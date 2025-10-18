from ml_util.spacy_interface import  SpacyHolder
from src.cpt_holder import RawCPT

spacy_holder = SpacyHolder.build(disable_modules=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"])


from collections import Counter, defaultdict
from ml_util.intertools_wrapper import powerset
from typing import List, FrozenSet, Dict, Iterable, Set, Tuple
import numpy as np

class UniCounter:
    def __init__(self,
                 *,
                 min_size: int = 1,
                 max_size: int = 2,
                 min_uni: int = 4,
                 lemma_stop_list: Iterable[str] = None):
        self.min_size = min_size
        self.max_size = max_size
        self.min_uni = min_uni
        self._lemma_vocab: List[str] = []
        self._tally_by_order: Dict[int, Counter[FrozenSet[int], int]] = defaultdict(Counter)
        self._tokens_seen: int = 0
        self._lemma_stop_list: Set[str] = set(lemma_stop_list) if lemma_stop_list is not None else set([])

        self._uni_log_probs = None
        self._usable_uni: np.ndarray[bool] = None

    @property
    def usable_uni(self) -> np.ndarray[bool]:
        if self._usable_uni is None:
            _ = self.uni_log_probs
        return self._usable_uni

    @property
    def uni_log_probs(self) -> np.ndarray[float]:
        if self._uni_log_probs is None:
            uni = np.array([self._tally_by_order[1][frozenset([i])] for i in range(len(self._lemma_vocab))])
            self._usable_uni = (uni >= self.min_uni)
            uni = -np.log(uni / float(self._tokens_seen))
            self._uni_log_probs = uni

        return self._uni_log_probs

    def _give_lemma_ind(self,
                        in_str: str) -> int:
        try:
            o = self._lemma_vocab.index(in_str)
        except ValueError:
            o = len(self._lemma_vocab)
            self._lemma_vocab.append(in_str)

        return o

    def add_sentence(self,
                     raw_in = Iterable[str]):
        self._tokens_seen += len(list(raw_in))
        by_ind = [self._give_lemma_ind(t) for t in set(raw_in) if t not in self._lemma_stop_list]

        for s in powerset(by_ind, min_cnt=self.min_size, max_cnt=self.max_size):
            self._tally_by_order[len(s)][frozenset(s)] += 1

    def find_observations(self):
        tokens_seen = float(self._tokens_seen)
        give_log_prob = lambda cnt: -np.log(cnt/tokens_seen)

        min_delta = 0.0

        uni_deltas: List[Tuple[FrozenSet[int], int]] = []
        for order in sorted(list([o for o in self._tally_by_order.keys() if o > 1]), reverse=True):
            for uni_set, raw_cnt in self._tally_by_order[order].items():
                if raw_cnt < 10 and self.usable_uni[list(uni_set)].sum() < len(uni_set):
                    continue
                comb_uni = self.uni_log_probs[list(uni_set)].sum()
                lp = give_log_prob(raw_cnt)
                delta = comb_uni - lp
                if abs(delta) > min_delta:
                    uni_deltas.append([uni_set, delta])

        uni_deltas = sorted(uni_deltas, key=lambda x: (-abs(x[1]), x[1], x[0]))
        pass


from ml_util.list_file import load_list_file
stop_list = load_list_file("stop_word_list.txt", min_len=1)
ortho_for_lemma = defaultdict(Counter)
uni_counter = UniCounter(min_size=1, max_size=2, lemma_stop_list=stop_list)
cpt_code_file = '../Consolidated_Code_List.txt'

raw_cpt_table = RawCPT(cpt_code_file, required_init_strings=['2'])

# s = "Anesthesia for open or surgical arthroscopic/endoscopic procedures on distal radius, distal ulna, wrist, or hand joints; not otherwise specified"

import re

five_d = re.compile(r"^[0-9]{5}$")

ready = [raw_cpt_table.value_for_cpt_field(cpt, 'Long') for cpt in raw_cpt_table.cpt_codes if five_d.match(cpt)]

for d in spacy_holder.run_pipe(ready):
    for s in d.sents:
        comb = [(str(t), t.lemma_) for t in s]
        for t in s:
            o, l = (str(t), t.lemma_)
            ortho_for_lemma[l][o] += 1
        uni_counter.add_sentence([c[1] for c in comb])

uni_counter.find_observations()

pass