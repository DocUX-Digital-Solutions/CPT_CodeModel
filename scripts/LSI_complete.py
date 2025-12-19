from ml_util.classes import ClassInventory
from ml_util.modelling.faiss_interface import SearchIndexWrapper
from ml_util.spacy_interface import  SpacyHolder
from src.cpt_holder import RawCPT
import faiss

spacy_holder = SpacyHolder.build(disable_modules=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"])


from collections import Counter, defaultdict
from ml_util.intertools_wrapper import powerset
from typing import List, FrozenSet, Dict, Iterable, Set, Tuple
import numpy as np
from frozendict import frozendict
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_array, coo_matrix, vstack, hstack

from ml_util.docux_logger import give_logger, configure_logger

logger = give_logger()

class Vocab:
    def __init__(self):
        self._voc: List[str] = []

    def __len__(self):
        return len(self._voc)

    def get_voc_ind(self,
                    ortho: str) -> int:
        try:
            return self._voc.index(ortho)
        except ValueError:
            ind = len(self._voc)
            self._voc.append(ortho)
            return ind


class CorpusTracker:
    idf_methods = ('smooth', 'double_normalization')
    def __init__(self,
                 spacy_holder: SpacyHolder,
                 *,
                 stop_list: List[str] = None,
                 idf_method: str = 'smooth',
                 match_method: str = 'cosine_similarity',
                 class_inventory: ClassInventory = None,
                 ):
        self._class_inventory = class_inventory
        self._match_method = match_method
        self._spacy_holder = spacy_holder
        if stop_list is None:
            stop_list = []
        self._stop_list = set(stop_list)

        assert idf_method in self.idf_methods
        self.idf_method = idf_method
        # self._vocab: List[str] = []
        self._vocab = Vocab()
        self._unweighted_doc_vecs: List[np.ndarray[int]] = []
        self._label_inds: List[int] = []
        self.label_inds: np.ndarray = None
        self._raw_docs: List[str] = []
        self._unweighted_doc_matrix: coo_matrix = None
        self._idf: np.ndarray[float] = None
        self._lex_type_weights: np.ndarray[np.float32] = None
        self._trunc_svd_by_dims: Dict = {}
        self._trunc_data_by_dims: Dict = {}
        self._index_by_dims: Dict[int, SearchIndexWrapper] = {}

    @property
    def unweighted_doc_matrix(self) -> coo_matrix:
        if self._unweighted_doc_matrix is None:
            voc_size = self.voc_size
            for d in self._unweighted_doc_vecs:
                d.resize((1, voc_size))
            self._unweighted_doc_matrix = vstack(self._unweighted_doc_vecs)
            # logger.info(f"unweighted_doc_matrix: {self._unweighted_doc_matrix.shape} "
            #             f"voc_size: {voc_size}")
            # save space
            self._unweighted_doc_vecs = None

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

    @property
    def lex_type_weights(self) -> np.ndarray[np.float32]:
        if self._lex_type_weights is None and self._class_inventory is None:
            self._lex_type_weights = self.idf
        elif self._lex_type_weights is None and self._class_inventory is not None:
            hier_weights: List[np.ndarray[np.float32]] = []
            hier_tf_weights: List[np.ndarray[np.float32]] = []
            for cs in self._class_inventory.hierarchical_segmentations:
                USE_TF = False
                logger.info(f"cs len: {cs.label_length}")
                mapped_down: coo_matrix = self.unweighted_doc_matrix.copy()
                logger.info(f"label_inds: {self.label_inds.shape} self.unweighted_doc_matrix.row.max(): {self.unweighted_doc_matrix.row.max()}")
                try:
                    # Need to deal with mapping to codes
                    mapped_down.row = cs.inverse[self.label_inds[self.unweighted_doc_matrix.row]]
                except:
                    raise
                row_cnt = mapped_down.row.max() + 1
                if  row_cnt < 2:
                    continue
                if self.idf_method == 'double_normalization':
                    # todense forces summation for shared cells.
                    mapped_down = mapped_down.todense()[:row_cnt]
                    # Normalize counts by the number of "documents" for each class.
                    mapped_down_tf = (mapped_down /
                          np.expand_dims(cs.counts, 1).repeat(mapped_down.shape[1], axis=1))
                    # mapped_down_tf =  1 + np.log(mapped_down_tf)
                    # Need to redistribute over the data points
                    mapped_down_idf = np.log(cs.size / (mapped_down_tf > 0).sum(axis=0))
                    weights = mapped_down_idf
                    if USE_TF:
                        tf_weights = (mapped_down_tf * mapped_down_idf).max(axis=0)
                else:
                    raise NotImplementedError

                hier_weights.append(weights)
                if USE_TF:
                    hier_tf_weights.append(tf_weights)

            if USE_TF:
                top_tf_weights = np.stack(hier_tf_weights).argmax(axis=0)
                top_coords = np.stack((np.arange(top_tf_weights.shape[0]), top_tf_weights)).T
                self._lex_type_weights = np.stack(hier_weights).T[top_coords[:, 0], top_coords[:, 1]]
            else:
                self._lex_type_weights = np.stack(hier_weights).max(axis=0)

        assert self._lex_type_weights is not None
        assert isinstance(self._lex_type_weights, np.ndarray) and len(self._lex_type_weights.shape) == 1

        return self._lex_type_weights

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
            loc_matrix = np.multiply(loc_matrix, self.lex_type_weights)

            curr = TruncatedSVD(n_components=dims)
            self._trunc_data_by_dims[dims] = curr.fit_transform(loc_matrix).astype(np.float32)
            self._trunc_svd_by_dims[dims] = curr
            self._index_by_dims[dims] = SearchIndexWrapper(self._trunc_data_by_dims[dims], similarity_measure=self._match_method)

        return curr

    # def lda_transform(self,
    #                   docs: str | List[str],
    #                   *,
    #                   dims: int = None) -> np.ndarray[float]:
    #     if dims is None:
    #         if len(self._trunc_svd_by_dims) == 1:
    #             dims = self._trunc_svd_by_dims.keys()[0]
    #         else:
    #             raise ValueError
    #
    #     encoded = self.encode_docs_unnormalized(docs)
    #     transformed = self._trunc_svd_by_dims[dims].transform(encoded)
    #
    #     return transformed

    # def get_voc_ind_and_add(self,
    #                         ortho: str) -> Tuple[int, bool]:
    #     try:
    #         return self._vocab.index(ortho), False
    #     except ValueError:
    #         ind = len(self._vocab)
    #         self._vocab.append(ortho)
    #         return ind, True
    #
    # def get_voc_ind(self,
    #                 ortho: str) -> int:
    #     return self.get_voc_ind_and_add(ortho)[0]
    #
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
                               raw_docs: str | List[str]) -> List[np.ndarray[float]]:
        docs = self._prep_docs(raw_docs)

        if isinstance(docs, str):
            docs = [docs]

        def get_doc_array(doc) -> np.ndarray[int]:
            tok_tallies = Counter(doc.strip().split())
            cnts = []
            loc_inds = []
            for k, v in tok_tallies.items():
                if k not in self._stop_list:
                    loc_inds.append(self._vocab.get_voc_ind(k))
                    cnts.append(v)
            cnts = np.array(cnts)
            return coo_array((cnts,
                              (np.zeros_like(cnts), np.array(loc_inds))),
                             (1, self.voc_size))

        return [get_doc_array(d) for d in docs]

    def add_docs(self,
                 docs: List[str],
                 label_inds: List[int],
                 ):
        if self._unweighted_doc_vecs is None:
            raise NotImplementedError

        self._raw_docs.extend(docs)

        self._unweighted_doc_vecs.extend(
            self.encode_docs_unweighted(docs)
        )
        self._label_inds.extend(label_inds)

    def search_docs(self,
                    docs: List[str],
                    svd_dims: int = 30,
                    ):
        logger.info(f"voc_size: {self.voc_size}")
        encoded = vstack(self.encode_docs_unweighted(docs)).todense().astype(np.float32)
        weighted = np.multiply(encoded, self.lex_type_weights)
        if self._match_method in ('cosine_similarity', 'double_normalization'):
            weighted = weighted.astype(np.float32)
            faiss.normalize_L2(weighted)
        trunc_svd = self.get_trunc_svd(svd_dims)
        transformed = trunc_svd.transform(weighted).astype(np.float32)

        # Need to search with faiss!!!
        distances, indices = self._index_by_dims[svd_dims].search(transformed)

        top_n = 10
        for doc, dist, inds in zip(docs, distances, indices):
            logger.info(f"doc: {doc}")
            for n in range(top_n):
                logger.info(f"{n}\t{dist[n]:.4f}\n{self._raw_docs[inds[n]]}")

from ml_util.list_file import load_list_file
stop_list = load_list_file("stop_word_list.txt", min_len=1)
# ortho_for_lemma = defaultdict(Counter)
cpt_code_file = '../Consolidated_Code_List.txt'

#raw_cpt_table = RawCPT(cpt_code_file, required_init_strings=['2'])
#raw_cpt_table = RawCPT(cpt_code_file, required_init_strings=['2'], digit_only=True)
raw_cpt_table = RawCPT(cpt_code_file, digit_only=True)
cpt_inventory = raw_cpt_table.give_inventory(min_form_count_per_class=1)

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

tracker.search_docs(toy, svd_dims=300)


pass