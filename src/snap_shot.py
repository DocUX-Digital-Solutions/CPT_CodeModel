from typing import List, Dict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from ml_util.modelling.batch_all import BatchAllDataset, give_ranges_by_common
from ml_util.classes import ClassInventory
from ml_util.modelling.faiss_interface import KMeansWeighted
from ml_util.modelling.searcher import SearchIndex
from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder

from ml_util.docux_logger import give_logger

logger = give_logger()

class EmbeddedBase:
    def __init__(self,
                 gpt_inventory: ClassInventory,
                 holder: SentenceTransformerHolder,
                 ):
        self.gpt_inventory = gpt_inventory
        self.strings, self.labels, self.string_inds = gpt_inventory.get_flat()
        self.space_size = self.labels.shape[0]
        self.embeddings = holder.encode_no_grad(self.strings)


class Searcher(EmbeddedBase):
    def __init__(self,
                 gpt_inventory: ClassInventory,
                 holder: SentenceTransformerHolder,
                 *,
                 n_nearest: int = 10,
                 ):
        super().__init__(gpt_inventory, holder)
        self.holder = holder
        self.search_index = SearchIndex(self.holder, self.embeddings, self.labels, n_nearest=n_nearest)

    def search(self,
               query: str) -> List[str]:
        return [self.gpt_inventory.get_label_with_description(label_ind, score=score)
                for label_ind, score in zip(*self.search_index.query_search(query))]


class SnapShot(EmbeddedBase):
    def __init__(self,
                 gpt_inventory: ClassInventory,
                 holder: SentenceTransformerHolder,
                 train_dataset: BatchAllDataset,
                 ):
        super().__init__(gpt_inventory, holder)
        # self.gpt_inventory = gpt_inventory
        # self.strings, self.labels, self.string_inds = gpt_inventory.get_flat()
        # self.space_size = self.labels.shape[0]
        # vecs = holder.encode_no_grad(self.strings)
        self.km = KMeansWeighted(self.embeddings,
                                 self.labels,
                                 np.array(give_ranges_by_common()),
                                 gpt_inventory,
                                 )

        self.embedding_sim_matrix = self.km.raw_similarity_matrix

        self.train_dataset = train_dataset
        self._train_mask: np.ndarray[bool] = None
        self._entry_similarity_matrix: np.ndarray[int] = None

    @property
    def train_inds(self) -> List[int]:
        return self.train_dataset.all_flat_inds

    @property
    def other_inds(self) -> List[int]:
        return [i for i in range(self.space_size) if i not in self.train_inds]

    @property
    def train_mask(self) -> np.ndarray[bool]:
        if self._train_mask is None:
            train_mask = np.zeros_like(self.labels, dtype=np.bool_)
            train_mask[self.train_dataset.all_flat_inds] = True
            train_mask = np.expand_dims(train_mask, axis=0).repeat(len(self.labels), axis=0)
            train_mask = train_mask * train_mask.T
            self._train_mask = train_mask

        return self._train_mask

    def _get_entry_similarity_matrix(self):
        by_row = np.repeat(np.expand_dims(self.labels, 0), self.labels.shape[0], axis=0)
        entry_similarity_matrix = self.gpt_inventory.label_similarity_matrix[by_row, by_row.T]
        self._entry_sim_values = np.unique(entry_similarity_matrix)
        # The max value is only for identity...
        self._top_val = self._entry_sim_values.max()
        self._entry_similarity_matrix = (
                entry_similarity_matrix +
                (torch.eye(entry_similarity_matrix.shape[0]) * (1 + self._entry_sim_values.max()))
        )

    @property
    def entry_similarity_matrix(self) -> np.ndarray[int]:
        if self._entry_similarity_matrix is None:
            self._get_entry_similarity_matrix()
        return self._entry_similarity_matrix

    @property
    def entry_sim_values(self) -> List[int]:
        if self._entry_sim_values is None:
            self._get_entry_similarity_matrix()
        return self._entry_sim_values

    @property
    def top_val(self) -> int:
        if self._top_val is None:
            self._get_entry_similarity_matrix()
        return self._top_val

    @property
    def sorted_sim_ranks(self) -> np.ndarray[int]:
        return self.entry_similarity_matrix.sort(axis=1, descending=True)[0]

    @property
    def cossim_ranks(self) -> np.ndarray[int]:
        # 1 for identity...
        return self.space_size - 1 - self.embedding_sim_matrix.argsort(axis=1).argsort(axis=1)

    def get_correlations(self,
                         *,
                         use_inds: List[int] = None):
        ind_prep = lambda m: m[use_inds] if use_inds is not None else m

        pearson = pearsonr(ind_prep(self.entry_similarity_matrix), ind_prep(self.embedding_sim_matrix), axis=1)
        spearman = spearmanr(ind_prep(self.entry_similarity_matrix), ind_prep(self.embedding_sim_matrix), axis=1)

        return pearson.correlation.mean(), spearman.correlation.mean()

    @property
    def top_sim_inds(self) -> torch.Tensor:
        return torch.argwhere(self.entry_similarity_matrix == self.top_val)

    def get_top_rank_analysis(self,
                              use_inds: List[int] = None):
        use_top_sim_inds = self.top_sim_inds if use_inds is None else self.top_sim_inds[use_inds]
        top_ranks = self.cossim_ranks[use_top_sim_inds.T[0], use_top_sim_inds.T[1]]

        denom = self.space_size if use_inds is None else len(use_inds)
        top = [f"{(top_ranks <= n).sum() / float(denom):.3f}" for n in range(1, 10)]

        return (f"top rank mean: {top_ranks.mean():.3f} ({top_ranks.std()}, {top_ranks.min()}-{top_ranks.max()})\t"
                f"top 10: {top}")

    def get_sim_errors(self) -> Dict:
        out = {}
        for split, loc_inds in (('train', self.train_inds), ('other', self.other_inds)):
            if len(loc_inds) < 1:
                continue
            for_cossim = torch.stack(
                [es[rc] for es, rc in
                 zip(self.entry_similarity_matrix[loc_inds], self.cossim_ranks[loc_inds] - 1)])
            diffs = (self.sorted_sim_ranks[loc_inds] - for_cossim)
            # Identity comes first...
            diffs[:, 0] = 0
            diffs = diffs.clip(min=0, max=100)
            diffs = diffs.unique(return_counts=True)
            out[split] = {
                'diffs': diffs,
                'str': f"{diffs[1]}\n"
                       f"{self.get_correlations(use_inds=loc_inds)}\n"
                       f"{self.get_top_rank_analysis(use_inds=loc_inds)}"}

        return out

    def give_train_masked(self, src: np.ndarray[bool],
                          *,
                          in_mask: np.ndarray[bool] = None):
        if in_mask is None:
            in_mask = 1
        train_inds = np.argwhere(in_mask * self.train_mask).T
        other_inds = np.argwhere(in_mask * ~self.train_mask).T

        return src[train_inds[0], train_inds[1]], src[other_inds[0], other_inds[1]]


    def compare_to_prev(self,
                        prev: 'SnapShot',
                        ):
        assert np.array_equal(self.labels, prev.labels)

        output = []

        prev_sim_error_d = prev.get_sim_errors()
        curr_sim_error_d = self.get_sim_errors()
        output.append(f"sim errors: {prev_sim_error_d['train']['diffs'][0]}"
                      f"\nprev train: {prev_sim_error_d['train']['str']}")
        if 'other' in prev_sim_error_d:
            output.append(f"prev other: {prev_sim_error_d['other']['str']}")
        output.append(f"curr train: {curr_sim_error_d['train']['str']})")
        if 'other' in curr_sim_error_d:
            output.append(f"curr other: {curr_sim_error_d['other']['str']}")

        give_loc_stat_str = lambda loc: f"{np.mean(loc):.3f} ({np.std(loc):.3f}, {loc.min():.3f}-{loc.max():.3f})"

        give_stat_str = lambda src, in_mask: (
            " ".join([f"{n}\t{give_loc_stat_str(loc)}"
                      for n, loc in zip(('train', 'other'), self.give_train_masked(src, in_mask=in_mask))
                      if loc is not None and loc.shape[0] > 0]))

        for sim in self.entry_sim_values:
            mask = self.entry_similarity_matrix.numpy() == sim
            output.append(f"sim: {sim} count: {mask.sum()}")
            for n, ranks in (('prev', prev.cossim_ranks), ('curr', self.cossim_ranks)):
                output.append(f"sim: {sim}\tranks {n}:\t{give_stat_str(ranks, mask)}")

            for n, all_sims in (('prev', prev.embedding_sim_matrix),
                                ('curr', self.embedding_sim_matrix),
                                ):
                output.append(f"sim: {sim}\tdiffs {n}:\t{give_stat_str(all_sims, mask)}")

        logger.info("\n" + "\n".join(output))
