from collections import defaultdict
from typing import List, Dict, Tuple

import torch

from data import TorchTensorHolder, NameSpaceValidator
from scripts.prepare_reports import FakeScores
from umls.graph_umls import UMLSCache


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
