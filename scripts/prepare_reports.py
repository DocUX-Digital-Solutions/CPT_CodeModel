import cProfile
import pstats
import re
import torch
import numpy as np
from collections import defaultdict, Counter

from typing import Dict, FrozenSet, List

from SectionSplitter import SectionSplitter
from ml_util.data import TorchTensorHolder, NameSpaceValidator
from ml_util.reports.report_preparer import ReportPreparer
from ml_util.scispcacy_interface import SciSpacyInterface
from ml_util.umls.graph_umls import UMLSCache
from ml_util.umls.quick_umls import QuickUMLS_Matcher, UMLS_Matcher
#from datasets.packaged_modules.json.json import ujson_dumps

from ml_util.BM25_interface import BM25Index
from ml_util.modelling.faiss_interface import give_SearchIndexWrapper
from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder, SentenceCrossEncoder
from ml_util.medspacy_interface import MedSpacyHolder
from scripts.embed_descriptions import CPT_embeddings
from src.report_holder import ReportHolder

from ml_util.docux_logger import give_logger, configure_logger, give_default_time

logger = None

# Create a new variant?
# Need to implement: give_final_score
class NullScoreHandler:
    def give_final_score(self, *args, **kwargs) -> float:
        return 0.0

class ScoreHandler:
    def __init__(self,
                 umls_cache: UMLSCache,
                 target_cpts: List[str],
                 retrieved_cpt_cuis: List[str],
                 ):
        self.umls_cache = umls_cache
        self._entries = defaultdict(dict)

        self._target_cpts = tuple(target_cpts)
        self.retrieved_cpt_cuis = tuple(retrieved_cpt_cuis)
        self._loc_cui_to_cpts = defaultdict(list)
        for cpt in self._target_cpts:
            if cpt not in umls_cache.code_to_cui:
                print(f"No cui for CPT: {cpt} '{self.umls_cache.cui_to_preferred_term.get(cpt, None)}'")
            else:
                for cui in umls_cache.code_to_cui.get(cpt, []):
                    self._loc_cui_to_cpts[cui].append(cpt)

    def can_add(self, n0, n1, dist, path) -> bool:
        raise NotImplementedError

    def add(self, n0, n1, dist, path):
        raise NotImplementedError

    def give_final_score(self, n0) -> float:
        raise NotImplementedError

    def get(self, n0, n1):
        try:
            return self._entries[n0][n1]
        except KeyError:
            raise
    ###

    def give_ref_matches(self,
                         cpt_id: str):
        loc_cpt_matches = self._loc_cui_to_cpts.get(cpt_id)
        if loc_cpt_matches:
            for_cpt = f"report_cpts: {' '.join(loc_cpt_matches)}"
        else:
            for_cpt = 'NO CPT MATCHES'
        return for_cpt

    def process_cpts_to_umls(self,
                             umls_ids: List[str],
                             ):
        raise NotImplementedError

from typing import Tuple

class VecScoreHandler(ScoreHandler):
    def __init__(self,
                 umls_cache: UMLSCache,
                 target_cpts: List[str],
                 retrieved_cpt_cuis: List[str],
                 accept_all: bool,
                 tensor_holder: TorchTensorHolder,
                 *args,
                 **kwargs,
                 ):
        super().__init__(umls_cache=umls_cache,
                         target_cpts=target_cpts,
                         retrieved_cpt_cuis=retrieved_cpt_cuis)
        self._entries: Dict[Tuple[int, int], float] = {}
        self.accept_all = accept_all
        self.tensor_holder = tensor_holder

        self._umls_name_space: NameSpaceValidator = None
        self._cpt_name_space: NameSpaceValidator = None

        self._score_tensor: torch.Tensor = None

        self._final_cpt_scores: Dict[str, float]  = {}

    # def can_add(self, n0, n1, dist, path):
    #     prev_dist = self._entries.get((n0, n1))
    #     return (prev_dist and prev_dist > dist) is False

    # def add(self, n0, n1, dist, path):
    #     self._entries[(n0, n1)] = dist

    def process_cpts_to_umls(self,
                             umls_ids: List[str],  # CPTs
                             ):
        (self._cpt_name_space, self._umls_name_space), self._score_tensor = (
            self.tensor_holder.give_scores(self.retrieved_cpt_cuis, umls_ids))

    def give_final_score(self, cpt_id: str,
                         ) -> float:
        if cpt_id not in self._final_cpt_scores:
            cpt_vec_inds = self._cpt_name_space.give_inds_for_names(
                self.umls_cache.code_to_cui[cpt_id]
            )
            self._final_cpt_scores[cpt_id] =  self._score_tensor[torch.Tensor(cpt_vec_inds)].mean().item()

        return self._final_cpt_scores[cpt_id]

###

class FakeScores:
    def __init__(self,
                 umls_cache: UMLSCache,
                 ):
        self.umls_cache = umls_cache
        self._prev: Dict[FrozenSet[str], float] = dict()

    def get(self, n0, n1) -> float:
        l = frozenset([n0, n1])
        out = self._prev.get(l)
        if out is None:
            inds = self.umls_cache.graph_holder.get_name_ind_tuple(n0, n1)
            out = self.umls_cache.graph_holder.fake_adamic_adar_between(*inds)
            self._prev[l] = out
        return out

    def present_dist(self, n0, n1, dist: float):
        fake = self.get(n0, n1)
        return f"d: {dist:.4f} ({fake:.4f} -> {dist - fake:+.4f})"

###

class NetworkScoreHandler(ScoreHandler):
    def __init__(self,
                 umls_cache: UMLSCache,
                 target_cpts: List[str],
                 retrieved_cpt_cuis: List[str],
                 edge_count_cutoff: int,
                 accept_all: bool,
                 *args, **kwargs,
                 ):
        super().__init__(umls_cache=umls_cache,
                         target_cpts=target_cpts,
                         retrieved_cpt_cuis=retrieved_cpt_cuis)
        self.edge_count_cutoff = edge_count_cutoff
        self.accept_all = accept_all
        self.fake_scores = FakeScores(self.umls_cache)

    def process_cpts_to_umls(self,
                             # sources: List[str], # document UMLS CUIs
                             umls_ids: List[str],  # CPTs
                             ):
        inputs = [self.umls_cache.graph_holder.filter_illegal_cuis(set(d))
                  for d in (umls_ids, self.retrieved_cpt_cuis)]
        print(f"input counts: {[len(i) for i in inputs]}")
        paths_to_cpts = (
            self.umls_cache.shortest_paths_between_cui_sets(*inputs,
                                                            cutoff=self.edge_count_cutoff))
        for umls_id, for_umls_id in paths_to_cpts.items():
            print(
                f"umls_id: {umls_id} "
                f"{self.umls_cache.cui_to_preferred_term.get(umls_id, 'NO DESCRIPTION FOUND!')}")
            skipped = []
            for cpt_id, (dist, path) in for_umls_id.items():
                code = self.umls_cache.cui_to_code[cpt_id]
                okay = True
                acceptable = True
                path_with_data = None
                if len(path) > 0:
                    path_with_data = self.umls_cache.give_path_with_data(umls_id, cpt_id, path, reverse=True)
                    okay = self.umls_cache.secondary_okay_path(path_with_data)
                    if len(path_with_data) == 1:
                        if self.umls_cache.rep_for_immediate(path_with_data[0]) is None:
                            acceptable = False
                            # skipped.append((dist, path_with_data))
                            # continue
                    else:
                        if not self.umls_cache.acceptable_feature_path(path_with_data):
                            acceptable = False
                            # skipped.append((dist, path_with_data))
                            # continue
                if not okay and not self.accept_all:
                    skipped.append((dist, path_with_data, acceptable))
                    continue

                if not self.can_add(code, umls_id, dist, path):
                    continue

                print(f"USE:\tacceptable: {acceptable}\td: {self.fake_scores.present_dist(umls_id, cpt_id, dist)}"
                      f"\t{self.give_ref_matches(cpt_id)} "
                      f"{self.umls_cache.show_path(path_with_data) if path_with_data else 'DIRECT'}")
                # paths_code_to_umls[code][umls_id] = path
                # score_handler.add(code, umls_id, path, dist)
                self.add(code, umls_id, dist, path)
            # for s in sorted(skipped, key=lambda x: (len(x), x[0].first_cui, x[0].second_cui)):
            for dist, s, acceptable in sorted(skipped, key=lambda x:
            (x[0], len(x[1]), x[1][0].first_cui, x[1][0].second_cui)):
                # okay = umls_cache.secondary_okay_path(s)
                print(f"SKIPPED:\t{self.fake_scores.present_dist(umls_id, cpt_id, dist)}\tacceptable: {acceptable}\t"
                      f"{self.give_ref_matches(s[0].first_cui)}\t"
                      f"{self. umls_cache.show_path(s)}")
                pass
            pass
        pass


class PathScoreHandler(NetworkScoreHandler):
    def can_add(self, n0, n1, dist, path) -> bool:
        for_code = self._entries.get(n0)
        if for_code:
            prev_path = for_code.get(n1)
            if prev_path and len(prev_path) <= len(path):
                return False

        return True

    def add(self, n0, n1, dist, path):
        self._entries[n0][n1] = path

    def give_final_score(self, n0) -> float:
        return float(sum([1 / (1 + len(v)) for v in self._entries[n0].values()]))


class DistanceScoreHandler(NetworkScoreHandler):
    def __init__(self,
                 decay_rate: float = 2.0,
                 *args,
                 **kwargs, ):
        super().__init__(*args, **kwargs)
        self._decay_rate = decay_rate

    def can_add(self, n0, n1, dist, path) -> bool:
        for_code = self._entries.get(n0)
        if for_code:
            prev = for_code.get(n1)
            if prev is not None and (prev[0] < dist or prev[1] < len(path)):
                return False

        return True

    def add(self, n0, n1, dist, path):
        self._entries[n0][n1] = (dist, len(path))

    def give_final_score(self, n0) -> float:
        # multi = lambda score, cnt: -np.log(score / float(cnt)) * ((1.0 / cnt) ** (1 / self._decay_rate))
        # multi = lambda score, cnt: -np.log(score / float(cnt))
        # single = lambda score: -np.log(score)
        return float(sum([1.0 / (1 + score) ** 1 / self._decay_rate
                          for score, cnt in self._entries[n0].values()]))

def main():
    import argparse
    import json


    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', type=str, required=True)
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument('--log_file_base', type=str, required=True)
    parser.add_argument('--log_level', type=str, default='debug')
    parser.add_argument("--embedding_dump_dir", type=str, default="scripts/cpt_work_justBase")
    parser.add_argument('--l2_norm', action='store_true')
    parser.add_argument('--impose_sent_split', action='store_true')
    parser.add_argument('--n_nearest', type=int, default=20)
    parser.add_argument('--model_name', type=str,
                        default="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb")
    parser.add_argument('--bm25_dir', type=str)
    parser.add_argument('--bm25_weight', type=float, default=0.25)
    parser.add_argument('--cross_encoder_model', type=str)
    parser.add_argument('--cross_trust_remote', action='store_true')
    parser.add_argument('--min_retain_portion', type=float, default=0.3333)
    parser.add_argument('--umls_cache_dir', type=str)
                        #default='/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/cache_2026AA')
    parser.add_argument('--quick_umls_threshold', type=float, default=0.7)
    parser.add_argument('--edge_count_cutoff', type=int, default=8)
    parser.add_argument('--scispacy_model', type=str)
    parser.add_argument('--skip_umls', action='store_true')
    parser.add_argument('--accept_all', action='store_true')
    parser.add_argument('--use_distances', action='store_true')
    parser.add_argument('--cui2vec_csv', type=str,
                        )#default="/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/cui2vec_pretrained.csv")
    parser.add_argument('--purge_skip_sections', action='store_true',
                        help="Ignore text in designated types of sections.")
    parser.add_argument('--merge_short_hr', action='store_true')
    parser.add_argument('--simple_merge_left', type=int, default=100)
    parser.add_argument('--simple_merge_right', type=int, default=200)
    args = parser.parse_args()
    global logger
    logger = give_logger()
    configure_logger(logger, log_file=f'{args.log_file_base}_{give_default_time()}', level=args.log_level)
    logger.info(f"args: {args}")

    input_jsonl = args.input_jsonl
    min_retain_portion = args.min_retain_portion

    cui2vec = None
    if args.cui2vec_csv:
        cui2vec = TorchTensorHolder(args.cui2vec_csv)

    bm25_index = None
    bm25_weight = args.bm25_weight
    if args.bm25_dir is not None:
        import os
        bm25_index = BM25Index.load(os.path.join(args.bm25_dir, 'bm25_index.json'))

    report_preparer = ReportPreparer(skip_sent_split=not args.impose_sent_split,
                                     need_merge_short_hr=args.merge_short_hr,
                                     simple_merge_left=args.simple_merge_left,
                                     simple_merge_right=args.simple_merge_right)
    by_odi: Dict[str, ReportHolder] = {}
    with open(input_jsonl, "r", encoding='utf-8') as in_H:
        for line in in_H:
            try:
                rh = ReportHolder.from_dict(json.loads(line.strip()))
            except:
                raise
            assert rh.operative_document_id not in by_odi
            by_odi[rh.operative_document_id] = rh
            # DIGDI
            # if len(by_odi) >= 10:
            #     break

    # medspacy_holder = MedSpacyHolder()
    section_splitter = SectionSplitter()
    if args.skip_umls:
        umls_matcher = None
    elif args.scispacy_model:
        umls_matcher = SciSpacyInterface(model_name=args.scispacy_model)
    else:
        umls_matcher = QuickUMLS_Matcher(threshold=args.quick_umls_threshold)

    if args.umls_cache_dir:
        umls_cache = UMLSCache.load(args.umls_cache_dir)

        class FakeScores:
            def __init__(self):
                self._prev: Dict[FrozenSet[str], float] = dict()

            def get(self, n0, n1) -> float:
                l = frozenset([n0, n1])
                out = self._prev.get(l)
                if out is None:
                    inds = umls_cache.graph_holder.get_name_ind_tuple(n0, n1)
                    out = umls_cache.graph_holder.fake_adamic_adar_between(*inds)
                    self._prev[l] = out
                return out

            def present_dist(self, n0, n1, dist: float):
                fake = self.get(n0, n1)
                return f"d: {dist:.4f} ({fake:.4f} -> {dist - fake:+.4f})"

        fake_scores = FakeScores()
    else:
        umls_cache = None
        fake_scores = None

    l2_norm = args.l2_norm
    n_nearest = args.n_nearest
    cpt_embeddings = CPT_embeddings.load(args.embedding_dump_dir, l2_norm=l2_norm)
    search_index = give_SearchIndexWrapper(cpt_embeddings.embeddings,
                                           similarity_measure='cosine_similarity' if l2_norm else 'ref_norm')
    model_holder = SentenceTransformerHolder.create(model_name=args.model_name)
    recall_by_cpt = defaultdict(Counter)
    cross_encoder = None if args.cross_encoder_model is None \
        else SentenceCrossEncoder.create(args.cross_encoder_model,
                                         trust_remote_code=args.cross_trust_remote)

    # ?? DIGDI
    def get_score_handler(*loc_args, **kwargs) -> ScoreHandler:
        if cui2vec:
            c = VecScoreHandler
        elif args.use_distances:
            c = DistanceScoreHandler
        else:
            c = PathScoreHandler

        return c(*loc_args, **kwargs)

    do_purge = args.purge_skip_sections
    inc_by_rank: List[np.ndarray] = []
    with open(args.output_jsonl, "w", encoding='utf-8') as out_H:
        PROFILE = False
        if True:
        #PROFILE = True
        # with cProfile.Profile() as pr:
            all_odi = sorted(list(by_odi.keys()))
            for odi in all_odi:
                logger.debug(f"odi: {odi}")
                raw_doc = by_odi[odi].pdf_text
                texts = []

                def give_simple():
                    spans_with_pre = [s for s in report_preparer.give_spans_with_pre(raw_doc)]
                    return [raw_doc[s.text_start:s.text_end]
                            for s in spans_with_pre]

                if not do_purge:
                    texts = give_simple()
                else:
                    offs, stretches = section_splitter.purge_skip_sections(raw_doc)
                    if sum([len(s) for s in stretches]) < min_retain_portion * len(raw_doc):
                        texts = give_simple()
                        logger.debug(f"Got less than {min_retain_portion} for: {odi}.")
                    else:
                        for off, stretch in zip(*section_splitter.purge_skip_sections(raw_doc)):
                            for s in report_preparer.give_spans_with_pre(stretch):
                                texts.append(raw_doc[off + s.text_start:off + s.text_end])

                doc_embeddings = torch.cat([b for b in model_holder.encode_no_grad(texts)])
                if l2_norm:
                    doc_embeddings = torch.nn.functional.normalize(doc_embeddings)
                distances, indices = search_index.search(doc_embeddings, n_nearest=n_nearest)
                if cross_encoder is not None:
                    new_sim = []
                    new_indices = []
                    for t_ind, t, in enumerate(texts):
                        cross_scores = cross_encoder.score_queries(
                            queries=[cpt_embeddings.descriptions[i] for i in indices[t_ind]],
                            candidate=t)
                        cs_as = (-1 * cross_scores).argsort()
                        new_sim.append(cross_scores[cs_as])
                        new_indices.append(indices[t_ind][cs_as])
                    distances, indices = [np.stack(l) for l in (new_sim, new_indices)]

                distances, ids, indices = [torch.tensor(a) for a in cpt_embeddings.id_compress(distances, indices)]

                class KeepMin:
                    def __init__(self):
                        self.value = 999999999999999

                    def add(self, v):
                        self.value = min(v, self.value)

                found_mins: Dict[int, KeepMin] = defaultdict(KeepMin)
                for id_a in ids:
                    for r, id in enumerate(id_a.tolist()):
                        found_mins[id].add(r)

                by_min = np.zeros((ids.shape[-1]), dtype=np.int16)
                for k, v in found_mins.items():
                    by_min[v.value] += 1
                for i in range(ids.shape[-1] -1, 0, -1):
                    by_min[i] = by_min[:i+1].sum()
                inc_by_rank.append(by_min)

                if bm25_index is not None:
                    d, i = bm25_index.search(texts)
                    bm_distances, bm_ids, bm_indices = [torch.tensor(a) for a in bm25_index.id_compress(d, i)]
                    distances, ids = \
                        cpt_embeddings.min_max_interpolate(
                            all_distances=[distances, bm_distances],
                            all_ids=[ids, bm_ids],
                            add_weights=[bm25_weight])

                target_cpts = sorted(list(set([pc.cpt4Code for pc in by_odi[odi].procedure_combinations])))

                if umls_matcher is not None:
                    # Integrate here!
                    index_matches = [
                        cpt_cui
                        for id in ids.unique().tolist() if id >= 0
                        for cpt_cui in umls_cache.code_to_cui.get(cpt_embeddings.label_for_id(id), [])]

                    # CACHE
                    score_handler = get_score_handler(umls_cache=umls_cache,
                                                      target_cpts=target_cpts,
                                                      retrieved_cpt_cuis=index_matches,  # Bring this through!!!
                                                      edge_count_cutoff=args.edge_count_cutoff,
                                                      accept_all=args.accept_all,
                                                      tensor_holder=cui2vec)

                    umls_matches = {t[0]: t for t in umls_matcher.give_matches(raw_doc).by_cui()}
                    score_handler.process_cpts_to_umls(list(umls_matches.keys()))
                else:
                    index_matches = [
                        cpt_embeddings.label_for_id(id)
                        for id in ids.unique().tolist() if id >= 0]

                    score_handler = NullScoreHandler()

                # Integrate -- ??? in place ??

                for pc in by_odi[odi].procedure_combinations:
                    first_desc, matches = cpt_embeddings.give_matches(pc.cpt4Code, ids)
                    # CACHE
                    loc_out = f"cpt: {pc.cpt4Code} '{first_desc}'"
                    if umls_cache:
                        loc_out += (f" final_score: {score_handler.give_final_score(pc.cpt4Code):.2f}"
                                    f" cuis: {umls_cache.code_to_cui.get(pc.cpt4Code, None)}")
                    logger.debug(loc_out)
                    if first_desc is None or len(matches) < 1:
                        logger.debug(f"no match for {pc.cpt4Code} in {odi}.")
                        recall_by_cpt[pc.cpt4Code][-1] += 1
                        # Show the top matches...
                    else:
                        by_rank = torch.argsort(matches[:, 1])
                        by_score = torch.argsort(-1 * torch.tensor(distances)[matches[:, 0], matches[:, 1]])
                        # Just take the mean???
                        by_combined = (by_rank + by_score).argsort()
                        # use_for_sort = by_combined
                        use_for_sort = by_rank
                        # if by_max
                        # use_for_sort = by_score
                        for m_rank, m in enumerate(matches[use_for_sort]):
                            if m_rank == 0:
                                recall_by_cpt[pc.cpt4Code][m[1].item()] += 1
                            logger.debug(f"{odi}\t{pc.cpt4Code} m: {m_rank} "
                                        f"rank for string: {m[1]} "
                                        f"dist: {distances[m[0], m[1]]:.3f} "
                                        f"{texts[m[0]]}")

                            for rank, (score, cpt_id) in enumerate(zip(distances[m[0]][:m[1]], ids[m[0]][:m[1]])):
                                label = cpt_embeddings.label_for_id(cpt_id)
                                logger.debug(f"\tover at {rank} ({score:.3f}): "
                                            f"{label} "
                                            f"umls_inv: {score_handler.give_final_score(label):.2f} "
                                            f"{cpt_embeddings.first_description_for_id(cpt_id)}")
                    pass
                if PROFILE:
                    stats = pstats.Stats(pr)
                    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)  # Prints top 10 bottlenecks
                pass

                # Need to check...

    digits_only = re.compile(r"^[0-9]{5}$")
    view_inds = (1, 3, 5, 10, 15, 20, 30, 40)
    by_type = {k: 0 for k in view_inds}
    by_instance = {k: 0 for k in view_inds}
    total_instances = 0
    for cpt_code, tally in sorted(recall_by_cpt.items()):
        if digits_only.match(cpt_code) is None:
            continue
        total = sum([v for v in tally.values()])
        logger.info(f"cpt: {cpt_code} total: {total}")
        total = float(total)
        total_instances += total
        fr = tally.get(-1, 0)
        logger.info(f"FR:\t{fr} ({fr/total:.2f}%)")
        cum_vals = []
        cum = 0
        for k, v in sorted(tally.items()):
            if k >= 0:
                cum += v
                cum_vals.append((k, cum))
        c_ind = -1
        for v_ind in view_inds:
            while c_ind + 1 < len(cum_vals) and cum_vals[c_ind + 1][0] < v_ind:
                c_ind += 1
            loc = 0 if c_ind == -1 else cum_vals[c_ind][1]
            logger.info(f"{v_ind}\t{loc} ({(100*loc) / total:.2f}%)")
            if loc > 0:
                by_type[v_ind] += 1
                by_instance[v_ind] += loc
    total_types = float(len(recall_by_cpt))

    logger.info(f"total_types: {total_types} total_instances: {total_instances}")

    logger.info(f"SUM by view_ind")
    for v_ind in view_inds:
        logger.info(f"{v_ind}"
                    f" types: {by_type[v_ind]} ({(100 * by_type[v_ind]) / total_types:.2f}%)"
                    f" instances: {by_instance[v_ind]} ({(100 * by_instance[v_ind]) / total_instances:.2f}%)")

    mid = len(inc_by_rank) // 2
    rank_mat = np.stack(inc_by_rank).T
    for rank, a in enumerate(rank_mat):
        mean = a.mean()
        std = a.std()
        median = np.sort(a)[mid]
        logger.info(f"rank: {rank}\t{mean:.2f} ({std:0.2f}) median: {median}")




if __name__ == '__main__':
    main()
