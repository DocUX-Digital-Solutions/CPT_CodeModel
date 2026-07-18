from ml_util.cpt_holder import get_raw_code_table
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


code_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/Consolidated_Code_List.txt"

raw_cpt= get_raw_code_table(code_file)
code_inventory = raw_cpt.give_inventory(min_form_count_per_class=2,
                                        name="CPT Inventory",
                                        max_similarity=3)

# targ_init = 4
targ_init = 2
common_init = []

from typing import List, Tuple, Set

# jarowinkler_cutoff = 0.85
jarowinkler_cutoff = 0.2
class StrPair:
    _jarowinkler_cutoff = jarowinkler_cutoff
    _jarowinkler_prefix_weight = 0.1
    @staticmethod
    def lower_rep(ind) -> str:
        return common_init[ind].representations[0].lower()

    def __init__(self,
                 ind_0: int,
                 ind_1: int):
        self.ind_0 = ind_0
        self.ind_1 = ind_1
        self.inds = (ind_0, ind_1)
        self.str_0, self.str_1 = (common_init[i].representations[0] for i in (ind_0, ind_1))
        self._sm = SequenceMatcher(None, self.lower_rep(ind_0), self.lower_rep(ind_1))
        self._ratio = None
        self._opcodes = None
        self._common_str = None
        self._common_len = None
        self._jarowinkler = None

    def __str__(self):
        return (f"{common_init[self.ind_0].label}\t{self.str_0}"
                f"\n\t{common_init[self.ind_1].label}\t{self.str_1}"
                f"\njw: {self.jarowinkler:.4f}\tratio: {self.ratio:.4f}\t{self.common_str}")

    def get_jarowinkler(self,
                        *,
                        score_cutoff=None,
                        prefix_weight=None) -> float:
        return jw.jarowinkler_similarity(
            self.str_0, self.str_1,
            score_cutoff=score_cutoff if score_cutoff is not None else self._jarowinkler_cutoff,
            prefix_weight=prefix_weight if prefix_weight is not None else self._jarowinkler_prefix_weight)

    @property
    def jarowinkler(self) -> float:
        if self._jarowinkler is None:
            self._jarowinkler = self.get_jarowinkler()
        return self._jarowinkler

    @property
    def ratio(self) -> float:
        if self._ratio is None:
            self._ratio = self._sm.ratio()

        return self._ratio

    @property
    def opcodes(self) -> List[Tuple]:
        if self._opcodes is None:
            self._opcodes = self._sm.get_opcodes()

        return self._opcodes

    @property
    def common_str(self) -> Tuple[str]:
        if self._common_str is None:
            self._common_str = tuple([self.str_0[t[1]:t[2]] for t in self.opcodes
                                      if t[0] == 'equal'])

        return self._common_str

    @property
    def common_len(self) -> int:
        if self._common_len is None:
            self._common_len = sum([len(s) for s in self.common_str])

class StrCluster:
    def __init__(self,
                 pairs: List[StrPair] = None):
        self._str_pairs = []
        self._inds: List[int] = None

        if pairs is None:
            pairs = []
        elif not isinstance(pairs, list):
            pairs = [pairs]

        for p in pairs:
            self.add(p)

    def intersects(self,
                   other_pair) -> bool:
        if self._inds is None:
            return False

        return len(set(self._inds) & set(other_pair.inds)) > 0

    def add(self,
            new_pair: StrPair):
        if self._inds is None:
            self._str_pairs = [new_pair]
            self._inds = list(new_pair.inds)
        else:
            for ind in new_pair.inds:
                if ind not in self._inds:
                    self._inds.append(ind)
            self._str_pairs.append(new_pair)

    def least_common(self) -> Tuple[str]:
        if len(self._str_pairs) < 1:
            raise ValueError
        sp = sorted(self._str_pairs, key=lambda x: x.get_jarowinkler(score_cutoff=0.0))
        return sp[0].common_str


def proc_shared_init():
    global common_init
    if len(common_init) < 2:
        return

    print(f"common_init: {common_init[0].label[:targ_init]}")
    codes = sorted([f"{m.label}\t{m.representations[0]}" for m in common_init])
    print("\n".join(codes) + "\n")

    str_pairs = []
    grid_jw = np.zeros([len(common_init), len(common_init)])
    for i, m in enumerate(common_init):
        for j in range(i + 1, len(common_init)):
            new_pair = StrPair(i, j)
            grid_jw[i, j] = new_pair.jarowinkler
            grid_jw[j, i] = new_pair.jarowinkler
            str_pairs.append(new_pair)

    str_clusters: List[StrCluster] = None
    for ind, str_pair in enumerate(sorted(str_pairs, key=lambda x: - x.jarowinkler)):
        if str_pair.jarowinkler < jarowinkler_cutoff:
            break
        print(f"{ind}\t{str_pair}")
        pass
        # if str_clusters is None:
        #     str_clusters = [
        #         StrCluster([str_pair])
        #     ]
        # else:

    pass

for m in code_inventory.members:
    if m.label[0] == '2' and m.label[-1].isdigit():
        if len(common_init) > 0 and common_init[0].label[:targ_init] != m.label[:targ_init]:
            proc_shared_init()
            common_init = []
        common_init.append(m)

proc_shared_init()

#
# play_member_cnt = len(play_members)
#
# q_sim = np.zeros([len(play_members), len(play_members)], dtype=np.float32)
#
# for i, m in enumerate(play_members):
#     for j in range(i+1, play_member_cnt):
#         q_sim[i, j] = SequenceMatcher(None,
#                                       m.representations[0].lower(),
#                                       play_members[j].representations[0].lower()).quick_ratio()
#
#
# rev_sorted_indices = (1.0 - q_sim).argsort(axis=None)
# rev_sorted_indices = np.unravel_index(rev_sorted_indices, q_sim.shape)

pass
