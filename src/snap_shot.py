from typing import List, Dict, Iterable, Tuple

import numpy as np
import torch

from ml_util.modelling.batch_all import BatchAllDataset, give_ranges_by_common
from ml_util.classes import ClassInventory
from ml_util.modelling.faiss_interface import KMeansWeighted
from ml_util.modelling.utils import to_tensor
from ml_util.modelling.random_projection import RandomProjection
from ml_util.modelling.searcher import SearchIndex
from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder

from ml_util.docux_logger import give_logger

logger = give_logger()


class EmbeddedBase:
    def __init__(self,
                 gpt_inventory: ClassInventory,
                 holder: SentenceTransformerHolder,
                 *,
                 batch_size: int = 128,
                 tranform: RandomProjection = None,
                 l2_normalize: bool = False,
                 ):
        self.gpt_inventory = gpt_inventory
        self.strings, self.labels, self.string_inds = gpt_inventory.get_flat()
        self.space_size = self.labels.shape[0]
        self._holder = holder
        self._embeddings: torch.Tensor = None
        self.batch_size = batch_size
        self.transform = tranform
        self.l2_normalize = l2_normalize

    def embedding_iter(self,
                       *,
                       batch_size: int = None,
                       l2_normalize: bool = None,
                       convert_to_tensor: bool = True,
                       ) -> Iterable[torch.Tensor]:
        if batch_size is None:
            batch_size = self.batch_size
        if l2_normalize is None:
            l2_normalize = self.l2_normalize
        for e in self._holder.encode_no_grad(self.strings, batch_size=batch_size, l2_norm=l2_normalize,
                                             convert_to_tensor=convert_to_tensor):
            yield e

    def get_embeddings(self,
                       *,
                       transform: RandomProjection = None,
                       batch_size: int = None,
                       l2_normalize: bool = None) -> torch.Tensor:
        if transform is None:
            transform = self.transform
        if batch_size is None:
            batch_size = self.batch_size
        if l2_normalize is None:
            l2_normalize = self.l2_normalize
        out = [to_tensor(transform.apply(e).cpu() if transform is not None else e)
               for e in self.embedding_iter(batch_size=batch_size, l2_normalize=l2_normalize,
                                            convert_to_tensor=bool(transform is not None))]
        return torch.cat(out)

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = self.get_embeddings()

        return self._embeddings

class Searcher(EmbeddedBase):
    def __init__(self,
                 gpt_inventory: ClassInventory,
                 holder: SentenceTransformerHolder,
                 *,
                 n_nearest: int = 10,
                 batch_size: int = 128,
                 tranform: RandomProjection = None,
                 l2_normalize: bool = False,
                 ):
        super().__init__(gpt_inventory, holder, batch_size=batch_size, tranform=tranform, l2_normalize=l2_normalize)
        self.holder = holder
        self.search_index = SearchIndex(self.holder, self.get_embeddings(), self.labels, n_nearest=n_nearest)

    def search(self,
               query: str) -> List[str]:
        return [self.gpt_inventory.get_label_with_description(label_ind, score=score)
                for label_ind, score in zip(*self.search_index.query_search(query))]


class SnapShot(EmbeddedBase):
    def __init__(self,
                 code_inventory: ClassInventory,
                 holder: SentenceTransformerHolder,
                 train_dataset: BatchAllDataset,
                 *,
                 batch_size: int = None,
                 transform: RandomProjection = None,
                 l2_normalize: bool = False,
                 ):
        super().__init__(code_inventory, holder, batch_size=batch_size, tranform=transform, l2_normalize=l2_normalize)
        # self.gpt_inventory = gpt_inventory
        # self.strings, self.labels, self.string_inds = gpt_inventory.get_flat()
        # self.space_size = self.labels.shape[0]
        # vecs = holder.encode_no_grad(self.strings)
        self.km = KMeansWeighted(self.get_embeddings(),
                                 self.labels,
                                 torch.tensor(give_ranges_by_common(code_inventory.max_similarity)),
                                 code_inventory,
                                 )

        self.embedding_sim_matrix = self.km.raw_similarity_matrix

        self.train_dataset = train_dataset
        self._train_mask: torch.Tensor = None
        self._entry_similarity_matrix: torch.Tensor = None
        self._cossim_ranks: torch.Tensor = None
        self._sorted_sim_ranks: torch.Tensor = None
        self._top_sim_inds: torch.Tensor = None
        self._entry_sim_values: torch.Tensor = None

    @property
    def train_inds(self) -> torch.Tensor:
        return self.train_dataset.all_flat_inds

    @property
    def other_inds(self) -> torch.Tensor:
        full = torch.arange(self.space_size)
        matches = (full.unsqueeze(1) == self.train_inds).any(dim=1)
        return full[~matches]

    @property
    def train_mask(self) -> torch.Tensor:
        if self._train_mask is None:
            train_mask = torch.zeros_like(self.labels, dtype=torch.bool)
            train_mask[self.train_dataset.all_flat_inds] = True
            train_mask = train_mask.expand(len(self.labels), -1)
            train_mask = train_mask * train_mask.T
            self._train_mask = train_mask

        return self._train_mask

    def _get_entry_similarity_matrix(self):
        by_row = self.labels.expand(self.labels.shape[0], -1)
        entry_similarity_matrix = self.gpt_inventory.label_similarity_matrix[by_row, by_row.T]
        self._entry_sim_values = entry_similarity_matrix.unique()
        # The max value is only for identity...
        self._top_val = self._entry_sim_values.max()
        eye_matrix = (torch.eye(entry_similarity_matrix.shape[0]) * (1 + self._entry_sim_values.max()))
        self._entry_similarity_matrix = entry_similarity_matrix + eye_matrix

    @property
    def entry_similarity_matrix(self) -> torch.Tensor:
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
    def sorted_sim_ranks(self) -> torch.Tensor:
        if self._sorted_sim_ranks is None:
            self._sorted_sim_ranks = self.entry_similarity_matrix.sort(dim=1, descending=True)[0]
        return self._sorted_sim_ranks

    @property
    def cossim_ranks(self) -> torch.Tensor:
        # 1 for identity...
        if self._cossim_ranks is None:
            self._cossim_ranks = self.space_size - 1 - self.embedding_sim_matrix.argsort(dim=1).argsort(axis=1)

        return self._cossim_ranks

    def get_correlations(self,
                         *,
                         use_inds: List[int] = None) -> Tuple[float, float]:
        ind_prep = lambda m: (m[use_inds] if use_inds is not None else m).view(-1)

        # pearson = pearsonr(ind_prep(self.entry_similarity_matrix), ind_prep(self.embedding_sim_matrix), axis=1)
        # spearman = spearmanr(ind_prep(self.entry_similarity_matrix), ind_prep(self.embedding_sim_matrix), axis=1)
        #
        # return pearson.correlation.mean(), spearman.correlation.mean()
        from torchmetrics.regression import SpearmanCorrCoef, PearsonCorrCoef
        loc = (ind_prep(self.entry_similarity_matrix), ind_prep(self.embedding_sim_matrix))

        return PearsonCorrCoef()(*loc).tolist(), SpearmanCorrCoef()(*loc).tolist()

    @property
    def top_sim_inds(self) -> torch.Tensor:
        if self._top_sim_inds is None:
            self._top_sim_inds = torch.argwhere(self.entry_similarity_matrix == self.top_val)
        return self._top_sim_inds

    def get_top_rank_analysis(self,
                              use_inds: torch.Tensor = None):
        use_top_sim_inds = self.top_sim_inds if use_inds is None else self.top_sim_inds[use_inds]
        top_ranks = self.cossim_ranks[use_top_sim_inds.T[0], use_top_sim_inds.T[1]].to(torch.float)

        denom = self.space_size if use_inds is None else use_inds.shape[0]
        top = [f"{(top_ranks <= n).sum() / float(denom):.3f}" for n in range(1, 10)]

        return (f"top rank mean: {top_ranks.mean():.3f} ({top_ranks.std():.3f}, {top_ranks.min()}-{top_ranks.max()})\t"
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

    def give_train_masked(self, src: torch.Tensor,
                          *,
                          in_mask: torch.Tensor = None):
        if in_mask is None:
            in_mask = 1
        train_inds = torch.argwhere(in_mask * self.train_mask).T
        other_inds = torch.argwhere(in_mask * ~self.train_mask).T

        return src[train_inds[0], train_inds[1]], src[other_inds[0], other_inds[1]]

    def __str__(self):
        output = []

        curr_sim_error_d = self.get_sim_errors()
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
            n = 'curr'
            output.append(f"sim: {sim}\tranks {n}:\t{give_stat_str(self.cossim_ranks, mask)}")
            output.append(f"sim: {sim}\tdiffs {n}:\t{give_stat_str(self.embedding_sim_matrix, mask)}")

        return "\n".join(output)

    def compare_to_prev(self,
                        prev: 'SnapShot',
                        ):
        assert torch.equal(self.labels, prev.labels)

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

        give_loc_stat_str = lambda loc: f"{loc.mean():.3f} ({loc.std():.3f}, {loc.min():.3f}-{loc.max():.3f})"

        give_stat_str = lambda src, in_mask: (
            " ".join([f"{n}\t{give_loc_stat_str(loc.to(torch.float))}"
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
