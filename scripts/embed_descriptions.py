import json

from networkx.classes import neighbors

from ml_util.modelling.huggingface_interface import HuggingFaceHolder, HuggingFaceEncoder
from src.cpt_holder import get_raw_code_table
import numpy as np
from difflib import SequenceMatcher
import jarowinkler as jw

'''
Notes:
* Take if the initial match ends in a semicolon and XX
* Take initial match (whole words) if XX (Truncate at ,)
* ?? 'Reattachment of ', 'cutoff 
* ('Impression and custom preparation of ', 'a', 'al prosthesis')  -- take whole words -- before semicolon
'''

from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder
import torch
import os
from typing import List, Dict, Tuple
import faiss
import igraph as ig
import leidenalg


class ByRes:
    dump_file_exts = {'res': 'res.npz',
                      'membership': 'membership.npz',
                      'centroids': 'centroids.npz',
                      'probs': 'probs.npz'}

    dump_file_dtypes = {'membership': np.int16,
                        'centroids': np.float32,
                        'probs': np.float32}


    def __init__(self,
                 res: float,
                 membership: np.ndarray,
                 centroids: np.ndarray,
                 probs: np.ndarray):
        self.res = res
        self.centroids = centroids
        self.norm_centroids = (centroids.T / np.linalg.norm(centroids, axis=1)).T
        self.probs = probs
        self.class_prob_sums = probs.sum(axis=1)
        self.membership = membership
        self.membership_counts = np.unique(membership, return_counts=True)[1]

    @staticmethod
    def give_dump_name_base(*,
                            l2_norm: bool = False):
        if l2_norm:
            return 'dict_by_res_dump_norm'
        else:
            return 'dict_by_res_dump'

    def closest_by_centroid(self,
                            centroid: np.ndarray) -> Tuple[int, float]:
        centroid /= np.linalg.norm(centroid)
        sims = np.squeeze(self.norm_centroids @ np.expand_dims(centroid, 0).T)

        max_ind = int(sims.argmax())
        return max_ind, sims[max_ind]

    def get_top_jaccard(self,
                        query: np.ndarray[int]) -> Tuple[int, float]:
        n = query.shape[0]
        types, cnts = np.unique(self.membership[query], return_counts=True)
        j = sorted(
            [(types[i], cnt /
              (n - cnt + self.membership_counts[types[i]])
              )
             for i, cnt in enumerate(cnts)],
            key=lambda x: (-x[1], x[0]))
        return j[0]

    @property
    def label_cnt(self) -> int:
        return self.probs.shape[0]

    @property
    def class_cnt(self) -> int:
        return self.probs.shape[1]

    def to_dict(self):
        return {'res': self.res,
                'membership': self.membership.tolist(),
                'centroids': self.centroids.tolist(),
                'probs': self.probs.tolist(),
                }

    @classmethod
    def from_json(cls,
                  dict: Dict):
        return cls(dict['res'],
                   *[np.array(dict[f]) for f in ('membership', 'centroids', 'probs')]
                   )

    @classmethod
    def from_json_dict(cls,
                       dump_dir: str,
                       *,
                       l2_norm: bool = False) -> Dict[float, 'ByRes']:
        path = os.path.join(dump_dir, ByRes.give_dump_name_base(l2_norm=l2_norm))
        if not ByRes._found_dump(path):
            raise ValueError(f"Failed to find: {path}")
        file_name_for_key = lambda k: '.'.join([path, ByRes.dump_file_exts[k]])
        vals = {}
        for k, ext in ByRes.dump_file_exts.items():
            with np.load(file_name_for_key(k)) as val:
                if len(val.files) == 1:
                    vals[k] = val[val.files[0]]
                else:
                    vals[k] = [val[f_id] for f_id in val.files]

        out = {res: cls(**{k: v[ind]
                           for k, v in vals.items()})
               for ind, res in enumerate(vals['res'])}

        return out

    @staticmethod
    def _found_dump(path: str) -> bool:
        file_name_for_key = lambda k: '.'.join([path, ByRes.dump_file_exts[k]])
        return all([os.path.exists(file_name_for_key(k))
            for k in ByRes.dump_file_exts.keys()])

    @staticmethod
    def dump_json_dict(dump_dir: str,
                       by_res: Dict[float, 'ByRes'],
                       *,
                       l2_norm: bool = False):

        ## What is wrong with 'centroids'?

        path = os.path.join(dump_dir, ByRes.give_dump_name_base(l2_norm=l2_norm))
        if ByRes._found_dump(path):
            raise ValueError(f"Already have: {path}")

        raw_out = {k: [] for k in ByRes.dump_file_exts.keys()}
        for k, v in sorted(by_res.items(), key=lambda x: x[0]):
            for field_name, targ in raw_out.items():
                raw = getattr(v, field_name)
                dtype = ByRes.dump_file_dtypes.get(field_name, None)
                if dtype is not None:
                    if str(raw.dtype) != dtype:
                        raw = raw.astype(dtype)
                targ.append(raw)

        file_name_for_key = lambda k: '.'.join([path, ByRes.dump_file_exts[k]])
        for k, v in raw_out.items():
            f_name = file_name_for_key(k)
            if k == 'res':
                np.savez(f_name, v)
            else:
                np.savez(f_name, *v)


class CPT_embeddings:
    text_json_name = "embedding_texts.json"
    embedding_pickle_name = 'cpt_embeddings.pt'
    no_norm_embedding_pickle_name = 'no_norm_cpt_embeddings.pt'
    dist_fill = -100
    index_fill = -1

    def __init__(self,
                 labels: List[str],
                 descriptions: List[str],
                 embeddings: torch.Tensor,
                 *,
                 softmax_temp: float = 0.05,
                 l2_norm: bool = True,
                 use_centroid=True,
                 use_jaccard=False,
                 ):
        assert len(labels) == len(descriptions) == embeddings.shape[0]
        self.raw_labels = tuple(sorted(list(set(labels))))
        self.label_ids = torch.tensor([self.raw_labels.index(l) for l in labels])
        # self.labels = labels
        self.descriptions = descriptions
        self.embeddings = embeddings
        self.softmax_temp = softmax_temp
        self.l2_norm = l2_norm
        self.use_centroid = use_centroid
        self.use_jaccard = use_jaccard

    def label_for_id(self,
                     id: int) -> str:
        return self.raw_labels[id]

    def id_for_label(self,
                     label: str) -> int:
        return self.raw_labels.index(label)

    def first_description_for_id(self,
                                 id: int) -> str:
        first_ind = torch.nonzero(self.label_ids == id)[0].item()
        return self.descriptions[first_ind]

    def indices_as_ids(self,
                       indices) -> torch.Tensor:
        return self.label_ids[torch.tensor(indices)]

    def uniq_sorted_descriptions(self,
                                 matches: torch.Tensor):
        _, index = np.unique(self.label_ids[matches[:,1]].numpy(), return_index=True)
        return matches[torch.tensor(index)]

    def id_compress(self,
                    raw_distances: np.ndarray,
                    raw_indices: np.ndarray):
        np_label_ids = self.label_ids.numpy()
        ready_dist = []
        ready_indices = []
        for chunk_dist, chunk_indices in zip(raw_distances, raw_indices):
            # Get index of the highest-ranking match for each label.
            use_inds = np.sort(np.unique(np_label_ids[chunk_indices], return_index=True)[1])
            ready_dist.append(chunk_dist[use_inds])
            ready_indices.append(chunk_indices[use_inds])

        max_len = max([d.shape[0] for d in ready_indices])
        out_dist = []
        out_indices = []
        for r_dist, r_indices in zip(ready_dist, ready_indices):
            pad_len = max_len - r_dist.shape[0]
            out_dist.append(
                np.concatenate([r_dist, np.full((pad_len), self.dist_fill, dtype=r_dist.dtype)])
            )
            out_indices.append(
                np.concatenate([r_indices, np.full((pad_len), self.index_fill, dtype=r_indices.dtype)])
            )
        out_dist = np.stack(out_dist)
        out_indices = np.stack(out_indices)
        out_ids = np_label_ids[out_indices]
        return out_dist, out_ids, out_indices



    @classmethod
    def  load(cls,
             use_dir: str,
             *,
             l2_norm: bool = True,
             prior_dir: str = None):
        embedding_file = cls.embedding_pickle_name if l2_norm else cls.no_norm_embedding_pickle_name
        embeddings = torch.load(os.path.join(use_dir if prior_dir is None else prior_dir, embedding_file))

        jsons = []
        with open(os.path.join(use_dir, cls.text_json_name), "r", encoding='utf-8') as in_H:
            for line in in_H:
                raw = json.loads(line.strip())
                jsons.append(raw)

        return cls(*jsons, embeddings, l2_norm=l2_norm)

    @staticmethod
    def cache(out_dir: str,
              *,
              code_file: str = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/Consolidated_Code_List.txt",
              sbert_model: str = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
              hf_model: str = None,
              aggregation_mode: str = None,
              max_seq_length: int = 512,
              prior_dir: str = None,
              use_all_representations: bool = False,
              task_type: str = 'CPT'
              ):

        immediate_json = os.path.join(prior_dir if prior_dir is not None else out_dir,
                                      CPT_embeddings.text_json_name)
        if os.path.exists(immediate_json):
            raise ValueError(f"{immediate_json} already exists!")
        if prior_dir is not None:
            raise ValueError
        out_json = os.path.join(out_dir, CPT_embeddings.text_json_name)

        os.makedirs(out_dir, exist_ok=False)

        raw_cpt= get_raw_code_table(code_file, task_type=task_type)
        code_inventory = raw_cpt.give_inventory(min_form_count_per_class=2,
                                                name="CPT Inventory",
                                                max_similarity=3)

        desc = []
        labels = []
        for m in code_inventory.members:
            if m.label[-1].isdigit():
                for r in (m.representations if use_all_representations else m.representations[:1]):
                    labels.append(m.label)
                    desc.append(r)

        st_holder = CPT_embeddings.give_encoder(sbert_model=sbert_model,
                                                hf_model=hf_model,
                                                aggregation_mode=aggregation_mode,
                                                max_seq_length=max_seq_length)
        embeddings = [e for e in st_holder.encode_no_grad(desc, l2_norm=False)]

        embeddings = torch.cat(embeddings)

        with open(out_json, "w", encoding='utf-8') as out_H:
            out_H.write(json.dumps(labels, ensure_ascii=False) + "\n")
            out_H.write(json.dumps(desc, ensure_ascii=False) + "\n")

        torch.save(embeddings, os.path.join(out_dir, CPT_embeddings.no_norm_embedding_pickle_name))

        embeddings = torch.nn.functional.normalize(embeddings)

        torch.save(embeddings, os.path.join(out_dir, CPT_embeddings.embedding_pickle_name))

    @staticmethod
    def give_encoder(*,
                     sbert_model: str = None,
                     hf_model: str = None,
                     aggregation_mode: str = None,
                     max_seq_length: int = 512,
                     ):
        if sbert_model is None and hf_model is None:
            raise ValueError

        if hf_model is not None:
            return HuggingFaceEncoder.create(model_name=hf_model,
                                             aggregation_mode=aggregation_mode,
                                             max_sequence_length=max_seq_length)
        else:
            return SentenceTransformerHolder.create(sbert_model)

    @property
    def dim(self):
        return self.embeddings.shape[1]

    def softmax(self,
                x: np.ndarray
                ):
        z = np.exp((x - x.max()) / self.softmax_temp)

        return z / z.sum()

    def get_loc_centroid(self,
                         class_inds: List[int]) -> torch.Tensor:
        centroid = self.embeddings[torch.tensor(class_inds, dtype=torch.int)].mean(dim=0)
        if self.l2_norm:
            centroid = torch.nn.functional.normalize(centroid)

        return centroid

    def give_confusion_classes(self,
                               *,
                               l2_norm: bool = False,
                               k=50,
                               resolutions: List[float] = [0.05]) -> Dict[float, ByRes]:
        loc_embeddings = self.embeddings.cpu().numpy()
        max_len = np.linalg.norm(loc_embeddings, axis=0).max()
        if l2_norm:
            loc_embeddings /= max_len
            max_len = 1.0

        def run_knn():
            index = faiss.IndexFlatIP(self.dim)
            index.add(loc_embeddings)

            similarities, neighbors = index.search(
                loc_embeddings,
                k
            )
            return similarities, neighbors

        similarities, neighbors = run_knn()

        def create_graph():
            edges = []
            weights = []

            for i in range(len(loc_embeddings)):
                for nbr, sim in zip(
                    neighbors[i,1:],
                    similarities[i,1:]
                ):
                    edges.append((i, int(nbr)))
                    weights.append(float(sim))

            g = ig.Graph(
                n=len(loc_embeddings),
                edges=edges,
                directed=False
            )

            g.es["weight"] = weights

            return g

        g = create_graph()

        def get_centroids(membership: np.ndarray) -> np.ndarray:
            centroids = []
            for cluster_id in np.unique(membership):
                members = np.where(
                    np.array(membership) == cluster_id
                )[0]
                centroid = loc_embeddings[members].mean(axis=0)
                if self.l2_norm:
                    centroid /= np.linalg.norm(centroid)
                centroids.append(centroid)

            centroids = np.stack(centroids)
            return centroids

        by_res = {}
        for res in resolutions:
            partition = leidenalg.find_partition(
                g,
                leidenalg.CPMVertexPartition,
                weights="weight",
                resolution_parameter=res * max_len)
            centroids = get_centroids(np.array(partition.membership))
            sims = loc_embeddings @ centroids.T
            probs = np.array([
                self.softmax(row)
                for row in sims
            ])
            by_res[res] = ByRes(res, np.array(partition.membership), centroids, probs)

        return by_res

    def show_options(self, d, by_res):
        def show_desc(res_id, ind):
            try:
                as_list = lambda x: x if isinstance(x, list) else [x]
                return [self.descriptions[i]
                        for i in as_list(np.argwhere(by_res[res_id].membership == ind).squeeze().tolist())]
            except:
                raise

        out = {}
        for k, v in d.items():
            try:
                out[k] = show_desc(k, v)
            except:
                raise
        return out

    def work_for_phrase(self,
                        by_res,
                        phrase: str):
        have_rotator = [ind for ind, d in enumerate(self.descriptions) if phrase in d]

        have_rotator_cent = self.get_loc_centroid(have_rotator).cpu().numpy()

        if self.use_centroid:
            closest = {k: v.closest_by_centroid(have_rotator_cent)
                       for k, v in by_res.items()}
        elif self.use_jaccard:
            closest = {k: v.get_top_jaccard(np.array(have_rotator))
                           for k, v in by_res.items()}
        else:
            raise ValueError
        closest_opts = self.show_options({k: v[0] for k, v in closest.items()}, by_res)

        return closest, closest_opts

    def work_vector(self,
                    vec,
                    by_res):
        if isinstance(vec, torch.Tensor):
            vec = vec.cpu().numpy()
        closest = {k: v.closest_by_centroid(vec)
                   for k, v in by_res.items()}
        closest_opts = self.show_options({k: v[0] for k, v in closest.items()}, by_res)

        return closest, closest_opts

toy_strings = ('rotator',
               'penetrating wound',
               'Reattachment of cutoff',
               'Free osteocutaneous flap with microvascular anastomosis',
               'Arthrocentesis, aspiration and/or injection, intermediate joint or bursa',
               'Autograft for spine surgery only (includes harvesting the graft',
               'Bone graft',
               'Biopsy',

               "Exploration of the patient's prior fusion site was performed at T12, L1, L2. ",
               "The fusion site clinically looked significantly better from an infection "
               "standpoint compared to a week ago. There was significantly less purulent "
               "material. ",
               "The wound grossly looked like it was responding well to the "
               "combination of surgical debridement as well as the antibiotics.",
               "Again noted was "
               "severe amounts of osteolysis consistent with deep chronic bone infection.",

               "Exploration of the patient's prior fusion site was performed at T12, L1, L2. "
               "The fusion site clinically looked significantly better from an infection "
               "standpoint compared to a week ago. There was significantly less purulent "
               "material. "
               "The wound grossly looked like it was responding well to the "
               "combination of surgical debridement as well as the antibiotics."
               "Again noted was "
               "severe amounts of osteolysis consistent with deep chronic bone infection."
               )


def main():
    sbert_model = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    #work_dir = 'cpt_work'
    work_dir = 'cpt_work_no_norm'
    ontology_dump_dir = None

    #sbert_model = "NeuML/pubmedbert-base-embeddings"
    #work_dir = "cpt_work_justBase"
    sbert_model = "pritamdeka/S-PubMedBert-MS-MARCO"
    work_dir = 'cpt_work_MS-MARCO'

    code_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/Consolidated_Code_List.txt"
    task_type = 'CPT'

    task_type = 'ClinicianCPT'
    code_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/ClinicianDescriptor.tsv"
    work_dir = 'clinical_cpt_MS-MARCO'



    if False:
        # last hidden state
        hf_model = "ncbi/MedCPT-Query-Encoder"
        work_dir='MedCPT-query'
        hf_model = 'ncbi/MedCPT-Article-Encoder'
        aggregation_mode = 'cls'
        max_seq_length = 512

        ontology_dump_dir = None
        hf_model = 'ncbi/MedCPT-Query-Encoder'
        work_dir = 'MedCPT-query'

    # work_dir = 'MedCPT-article'
    # ontology_dump_dir = 'MedCPT-query'
    # hf_model = 'ncbi/MedCPT-Article-Encoder'

    # aggregation_mode = 'cls'
    max_seq_length = 512

    l2_norm = False
    # If add normalization, the clustering is not good, at all.
    # l2_norm = True
    try:
        if True:
            CPT_embeddings.cache(work_dir, sbert_model=sbert_model, code_file=code_file,
                                 task_type=task_type,
                                 use_all_representations=(task_type == 'ClinicianCPT'))
        else:
            CPT_embeddings.cache(work_dir,
                                 hf_model=hf_model,
                                 aggregation_mode=aggregation_mode,
                                 max_seq_length=max_seq_length,
                                 prior_dir=ontology_dump_dir)
    except ValueError:
        print(f"Will load from cache dir.")

    cpt_embeddings = CPT_embeddings.load(work_dir, l2_norm=l2_norm,
                                         prior_dir=ontology_dump_dir)
    try:
        by_res = ByRes.from_json_dict(work_dir, l2_norm=l2_norm)
    except ValueError:
        res = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, .7, 0.9, 0.95, 0.99]
        # res = [.1, .25, .5, .6, .7, .8, .9]
        by_res = cpt_embeddings.give_confusion_classes(resolutions=res, l2_norm=l2_norm)
        ByRes.dump_json_dict(work_dir, by_res, l2_norm=l2_norm)

    top_opts = lambda: closest_opts[
        sorted([(k, v) for k, v in closest.items()],
               key=lambda x: (-x[1][1], x[0]))[0][0]
    ]

    if True:
        for phrase in toy_strings[:8]:
            closest, closest_opts = cpt_embeddings.work_for_phrase(by_res, phrase)
            to = top_opts()
            print(f"phrase: {phrase}\ntop_opts ({len(to)}): {to}\n")
            pass
    else:
        for m_name in ("ncbi/MedCPT-Query-Encoder", 'ncbi/MedCPT-Article-Encoder'):
            query_encode = CPT_embeddings.give_encoder(hf_model=m_name,
                                                       aggregation_mode=aggregation_mode,
                                                       max_seq_length=max_seq_length)
        # if True:
        #     m_name = sbert_model
        #     query_encode = CPT_embeddings.give_encoder(sbert_model=sbert_model,
        #                                                max_seq_length=max_seq_length)
            q_encode = torch.cat(
                [b for b in
                 query_encode.encode_no_grad(toy_strings, batch_size=128, l2_norm=l2_norm)]
            )
            for q, v in zip(toy_strings, q_encode):
                closest, closest_opts  = cpt_embeddings.work_vector(v, by_res)
                to = top_opts()
                print(f"{m_name} phrase: {q}\ntop_opts ({len(to)}): {to}\n")
                pass

    pass

if __name__ == '__main__':
    main()
