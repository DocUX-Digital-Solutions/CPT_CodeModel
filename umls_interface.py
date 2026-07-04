from dataclasses import dataclass
from typing import Dict, List, Iterable
from collections import defaultdict


@dataclass(frozen=True)
class SemCategoryEntry:
    short_type: str
    long_type: str
    code: str
    description: str


class SemGroups:
    def __init__(self,
                 *,
                 raw_file="/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/SemGroups.txt"):
        self.by_code: Dict[str, SemCategoryEntry] = {}

        by_short: Dict[str, List[SemCategoryEntry]] = defaultdict(list)
        with open(raw_file, "r", encoding='utf-8') as in_H:
            for line in in_H:
                parts = line.strip().split('|')
                entry = SemCategoryEntry(*parts)
                self.by_code[entry.code] = entry
                by_short[entry.short_type].append(entry)

        self.by_short: Dict[str, Iterable[SemCategoryEntry]] = {k: tuple(v) for k, v in by_short.items()}

    def class_codes_for_code(self,
                             code: str) -> List[str]:
        return \
            [e.code
             for e in self.by_short[
                 self.by_code[code].short_type
             ]
             ]
