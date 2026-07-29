from ml_util.classes import ClassInventory
from ml_util.modelling.faiss_interface import SearchIndexWrapper
from ml_util.spacy_interface import  SpacyHolder
from ml_util.cpt_holder import RawCPT
import faiss

spacy_holder = SpacyHolder.build(disable_modules=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"])


from collections import Counter, defaultdict
from ml_util.intertools_wrapper import powerset
from typing import List, FrozenSet, Dict, Iterable, Set, Tuple, Any
import numpy as np
from frozendict import frozendict
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_array, coo_matrix, vstack, hstack

from ml_util.docux_logger import give_logger, configure_logger

from ml_util.list_file import load_list_file


logger = give_logger()

class Vocab:
    def __init__(self):
        self._items: Dict[str, Any] = {}
        self._lookup: List[str] = []

    def __len__(self):
        return len(self._lookup)

    def get_voc_ind(self,
                    item: Any) -> int:
        try:
            return self._items[item]
        except KeyError:
            ind = len(self._lookup)
            self._items[item] = ind
            self._lookup.append(item)

            return ind


class InventoryTracker:
    idf_methods = ('smooth', 'double_normalization')

    def __init__(self,
                 *,
                 idf_method: str = 'smooth',
                 match_method: str = 'cosine_similarity',
                 class_inventory: ClassInventory = None,
                 use_tf: bool = False,
                 ):
        self._class_inventory = class_inventory
        self._match_method = match_method
        assert idf_method in self.idf_methods
        self.idf_method = idf_method
        self.use_tf = use_tf

        self._vocab = Vocab()
        self._unweighted_entry_vecs: List[coo_array] = []
        self._label_inds: List[int] = []
        self.label_inds: np.ndarray = None
        self._unweighted_doc_matrix: coo_matrix = None
        self._idf: np.ndarray[float] = None
        self._type_weights: np.ndarray[np.float32] = None
        self._trunc_svd_by_dims: Dict = {}
        self._trunc_data_by_dims: Dict = {}
        self._index_by_dims: Dict[int, SearchIndexWrapper] = {}

    @property
    def unweighted_doc_matrix(self) -> coo_matrix:
        if self._unweighted_doc_matrix is None:
            voc_size = self.voc_size
            if len(self._label_inds) != len(self._unweighted_entry_vecs):
                raise ValueError
            for d in self._unweighted_entry_vecs:
                d.resize((1, voc_size))
            self._unweighted_doc_matrix = vstack(self._unweighted_entry_vecs)
            # logger.info(f"unweighted_doc_matrix: {self._unweighted_doc_matrix.shape} "
            #             f"voc_size: {voc_size}")
            # save space
            self._unweighted_entry_vecs = None

            self.label_inds = np.array(self._label_inds, dtype=np.int32)
            logger.info(f"unweighted_doc_matrix: {self._unweighted_doc_matrix.shape} "
                        f"row max: {self._unweighted_doc_matrix.row.max()} "
                        f"voc_size: {voc_size} label_inds {self.label_inds.shape} "
                        f"unique: {np.unique(self.label_inds).shape[0]}")

        return self._unweighted_doc_matrix

    @property
    def idf(self) -> np.ndarray[float]:
        if self._idf is None:
            raw_df = (self.unweighted_doc_matrix > 0).sum(axis=0)
            doc_cnt = self.unweighted_doc_matrix.shape[0]
            if self.idf_method == 'smooth':
                self._idf = np.log(doc_cnt / (1 + raw_df)) + 1
            else:
                raise NotImplementedError

        return self._idf

    def _get_chi_type_weights(self):
        from scipy.stats import chi2

        lex_totals = self.unweighted_doc_matrix.todense().sum(axis=0)
        hier_weights: List[np.ndarray[np.float32]] = []
        hier_tf_weights: List[np.ndarray[np.float32]] = []
        hier_chi2_p: List[np.ndarray[np.float32]] = []
        for cs in self._class_inventory.hierarchical_segmentations:
            expected = np.outer((cs.counts / cs.counts.sum()).numpy(), lex_totals)
            # logger.info(f"cs len: {cs.label_length}")
            mapped_down: coo_matrix = self.unweighted_doc_matrix.copy()
            # logger.info(
            #     f"label_inds: {self.label_inds.shape} self.unweighted_doc_matrix.row.max(): {self.unweighted_doc_matrix.row.max()}")
            mapped_down.row = cs.inverse[self.label_inds[self.unweighted_doc_matrix.row]]
            row_cnt = mapped_down.row.max() + 1
            if row_cnt < 2:
                continue # No contrast
            if self.idf_method == 'double_normalization':
                # todense forces summation for shared cells.
                mapped_down = mapped_down.todense()[:row_cnt]
                z = np.sum(np.power(mapped_down - expected, 2) / expected, axis=0)
                hier_chi2_p.append(
                    chi2.sf(z, -1 + row_cnt)
                )
                # Normalize counts by the number of "documents" for each class.
                mapped_down_tf = (mapped_down /
                                  np.expand_dims(cs.counts, 1).repeat(mapped_down.shape[1], axis=1))
                # mapped_down_tf =  1 + np.log(mapped_down_tf)
                # Need to redistribute over the data points
                mapped_down_idf = np.log(cs.size / (mapped_down_tf > 0).sum(axis=0))
                weights = mapped_down_idf
                if self.use_tf:
                    tf_weights = (mapped_down_tf * mapped_down_idf).max(axis=0)
            else:
                raise NotImplementedError

            hier_weights.append(weights)
            if self.use_tf:
                hier_tf_weights.append(tf_weights)

        if self.use_tf:
            raise NotImplementedError
            # This needs to be fixed...
            top_tf_weights = np.stack(hier_tf_weights).argmax(axis=0)
            top_coords = np.stack((np.arange(top_tf_weights.shape[0]), top_tf_weights)).T
            type_weights = np.stack(hier_weights).T[top_coords[:, 0], top_coords[:, 1]]
        else:
            type_weights = np.stack(hier_weights)


        p_thresh = 0.0001
        hier_chi2_p = np.stack(hier_chi2_p).T
        thresh_filter = hier_chi2_p <= p_thresh
        use_index = thresh_filter.argmin(axis=1)
        # use the highest-resolution counts if none are significant
        miss_inds = np.argwhere(~thresh_filter.any(axis=1)).flatten()
        use_index[miss_inds] = hier_chi2_p[miss_inds].argmin(axis=1)
        # thresh_voc = [self._vocab._voc[i] for i in np.argwhere(thresh_filter).squeeze().tolist()]
        # lowest_ind = hier_chi2_p.argmin(axis=0)
        type_weights = type_weights[use_index, np.arange(use_index.shape[0])]
        assert type_weights is not None
        assert isinstance(type_weights, np.ndarray) and len(type_weights.shape) == 1

        return type_weights

    @property
    def type_weights(self) -> np.ndarray[np.float32]:
        if self._type_weights is None:
            if self._class_inventory is None:
                self._type_weights = self.idf
            else:
                if self.idf_method in {'double_normalization'}:
                    self._type_weights = self._get_chi_type_weights()

        return self._type_weights

    @property
    def voc_size(self) -> int:
        return len(self._vocab)

    def get_trunc_svd(self,
                      dims: int) -> TruncatedSVD:
        curr = self._trunc_svd_by_dims.get(dims)
        if curr is None:
            #loc_matrix = self.doc_matrix
            # get tf/idf (or just idf...)
            # loc_matrix = loc_matrix.multiply(self.idf)
            loc_matrix = (self.unweighted_doc_matrix.todense() > 0).astype(np.float32)
            loc_matrix = np.multiply(loc_matrix, self.type_weights)

            curr = TruncatedSVD(n_components=dims)
            self._trunc_data_by_dims[dims] = curr.fit_transform(loc_matrix).astype(np.float32)
            self._trunc_svd_by_dims[dims] = curr
            self._index_by_dims[dims] = SearchIndexWrapper(self._trunc_data_by_dims[dims], similarity_measure=self._match_method)

        return curr

    def encode_item(self,
                    counts: List[float],
                    loc_inds: List[int]):
        assert len(loc_inds) < 1 or max(loc_inds) < self.voc_size

        counts = np.array(counts)
        encoded = coo_array((counts,
                            (np.zeros_like(counts), np.array(loc_inds))),
                           (1, self.voc_size))

        return encoded

    def add_unweighted_counts(self,
                              *,
                              counts: List[float] = None,
                              loc_inds: List[int] = None,
                              encoded: None | coo_array | List[coo_array]):
        if encoded is not None:
            if not isinstance(encoded, list):
                to_add = [encoded]
            else:
                to_add = encoded
        else:
            to_add = [self.encode_item(counts, loc_inds)]
        self._unweighted_entry_vecs.extend(to_add)

        return to_add

    def search_items(self,
                     encoded: List[coo_array],
                     svd_dims: int,
                     ) -> Tuple[np.ndarray[float], np.ndarray[int]]:
        encoded = vstack(encoded).todense().astype(np.float32)
        weighted = np.multiply(encoded, self.type_weights)
        if self._match_method in ('cosine_similarity', 'double_normalization'):
            weighted = weighted.astype(np.float32)
            faiss.normalize_L2(weighted)
        trunc_svd = self.get_trunc_svd(svd_dims)
        transformed = trunc_svd.transform(weighted).astype(np.float32)

        # Need to search with faiss!!!
        distances, indices = self._index_by_dims[svd_dims].search(transformed)

        return distances, indices


class CorpusTracker(InventoryTracker):
    def __init__(self,
                 spacy_holder: SpacyHolder,
                 *,
                 stop_list: List[str] = None,
                 idf_method: str = 'smooth',
                 match_method: str = 'cosine_similarity',
                 class_inventory: ClassInventory = None,
                 ):
        super().__init__(idf_method=idf_method,
                         match_method=match_method,
                         class_inventory=class_inventory)
        self._spacy_holder = spacy_holder
        if stop_list is None:
            stop_list = []
        self._stop_list = set(stop_list)
        self._raw_docs: List[str] = []

    @property
    def lex_type_weights(self) -> np.ndarray[np.float32]:
        return self.type_weights

    def _prep_docs(self,
                   raw_docs: str | List[str]):
        if isinstance(raw_docs, str):
            raw_docs = [raw_docs]

        prepped_docs = []
        for d in spacy_holder.run_pipe(raw_docs):
            prepped_docs.append(
                ' '.join([t.lemma_ for s in d.sents
                          for t in s])
                    )

        assert len(prepped_docs) == len(raw_docs)
        return prepped_docs

    def encode_docs_unweighted(self,
                               raw_docs: str | List[str],
                               ) -> List[coo_array]:
        docs = self._prep_docs(raw_docs)

        if isinstance(docs, str):
            docs = [docs]

        out = []
        for doc in docs:
            tok_tallies = Counter(doc.strip().split())
            cnts = []
            loc_inds = []
            for k, v in tok_tallies.items():
                if k not in self._stop_list:
                    loc_inds.append(self._vocab.get_voc_ind(k))
                    cnts.append(v)
            encoded = self.encode_item(cnts, loc_inds)
            out.append(encoded)

        return out

    def add_docs(self,
                 docs: List[str],
                 label_inds: List[int],
                 ):
        if self._unweighted_entry_vecs is None:
            raise NotImplementedError

        self._raw_docs.extend(docs)

        self.add_unweighted_counts(encoded=self.encode_docs_unweighted(docs))
        self._label_inds.extend(label_inds)

    def search_docs(self,
                    docs: List[str],
                    svd_dims: int = 30,
                    ):
        logger.info(f"voc_size: {self.voc_size}")
        encoded = self.encode_docs_unweighted(docs)
        distances, indices = self.search_items(encoded, svd_dims)

        top_n = 10
        for doc, dist, inds in zip(docs, distances, indices):
            logger.info(f"doc: {doc}")
            for n in range(top_n):
                logger.info(f"{n}\t{dist[n]:.4f}\t{self._raw_docs[inds[n]]}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop_list', type=str, default="stop_word_list.txt")
    parser.add_argument('--cpt_code_file', type=str, default='../Consolidated_Code_List.txt')
    args = parser.parse_args()
    stop_list = load_list_file(args.stop_list, min_len=1)
    # ortho_for_lemma = defaultdict(Counter)
    cpt_code_file = args.cpt_code_file

    #raw_cpt_table = RawCPT(cpt_code_file, required_init_strings=['2'])
    #raw_cpt_table = RawCPT(cpt_code_file, required_init_strings=['2'], digit_only=True)
    raw_cpt_table = RawCPT(cpt_code_file, digit_only=True)
    cpt_inventory = raw_cpt_table.give_inventory(min_form_count_per_class=1, max_similarity=5)

    # s = "Anesthesia for open or surgical arthroscopic/endoscopic procedures on distal radius, distal ulna, wrist, or hand joints; not otherwise specified"

    # f = 'Long'
    # # for f in ('Long', 'Consumer')
    # ready = [(cpt, raw_cpt_table.value_for_cpt_field(cpt, f))
    #          for cpt in raw_cpt_table.cpt_codes if self.five_d.match(cpt)
    #          ]

    ready = []
    label_inds = []
    for s, label_ind, string_ind in zip(*cpt_inventory.get_flat()):
        if string_ind == 0:
            ready.append(s)
            label_inds.append(label_ind)

    logger.info(f"label_inds: {len(label_inds)}")

    configure_logger(logger, log_file='lsi_complete.log')
    # tracker = CorpusTracker(spacy_holder, stop_list=stop_list, match_method='just_dot')
    tracker = CorpusTracker(spacy_holder, stop_list=stop_list,
                            idf_method='double_normalization',
                            class_inventory=cpt_inventory,
                            )
    tracker.add_docs(ready, label_inds)
    # print(f"unweighted_doc_matrix: {tracker.unweighted_doc_matrix.shape}")

    toy = ["rotator cuff repair", "femur fracture", "rotator cuff arthroscopy", "rotator cuff"]

    tracker.search_docs(toy, svd_dims=2000)


if __name__ == "__main__":
    main()