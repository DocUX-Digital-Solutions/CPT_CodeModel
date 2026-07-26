from quickumls import QuickUMLS
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, FrozenSet, Tuple
from collections import defaultdict


@dataclass(frozen=True)
class QuickMatch:
    cui: str
    preferred: int
    semtypes: FrozenSet[str]
    similarity: float
    term: str

    @classmethod
    def create(cls,
               raw: Dict):
        return cls(
            **{k: raw[k] for k in
               ('cui', 'preferred', 'similarity', 'term')
               },
            semtypes=frozenset(raw['semtypes'])
        )

@dataclass(frozen=True)
class QuickMatchSpan:
    start: int
    end: int
    ngram: str

    @classmethod
    def create(cls,
               raw: Dict):
        return cls(**{k: raw[k]
                      for k in ('start', 'end', 'ngram')})


@dataclass(frozen=True)
class QuickMatchSet:
    matches: Dict[QuickMatch, List[QuickMatchSpan]]

    @classmethod
    def create(cls,
               raw: List[List[Dict]]):
        loc = defaultdict(list)
        for loc_raw in raw:
            s = QuickMatchSpan.create(loc_raw[0])
            for l in loc_raw:
                loc[QuickMatch.create(l)].append(s)

        return cls(dict(loc))

    def by_match(self) -> Iterable[Tuple[QuickMatch, List[QuickMatchSpan]]]:
        for k in sorted(self.matches.items(), key=lambda x: (x[0].cui, -x[0].similarity, x[0].term)):
            yield k

    def by_cui(self) -> List[Tuple[str, List[Tuple[QuickMatch, List[QuickMatchSpan]]]]]:
        out_cuis = []
        for_cui = []
        for k, v in self.by_match():
            try:
                ind = out_cuis.index(k.cui)
            except ValueError:
                ind = len(out_cuis)
                out_cuis.append(k.cui)
                for_cui.append([])
            for_cui[ind].append((k, v))

        return [(c, fc) for c, fc in zip(out_cuis, for_cui)]


class QuickUMLS_Matcher:
    def __init__(self,
                 source_dir: str = '/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/2026AA/quickumls',
                 *,
                 threshold: float = 0.7,
                 similarity_name: str = 'jaccard'):
        self.matcher = QuickUMLS(source_dir, threshold=threshold, similarity_name=similarity_name)

    def give_matches(self,
                     query: str,
                     *,
                     best_match: bool = False,
                     top_n: int = None) -> QuickMatchSet:
        if top_n is not None and top_n == 1:
            best_match =True
        matches = self.matcher.match(query, best_match=best_match)
        if top_n is not None:
            matches = matches[:top_n]

        return QuickMatchSet.create(matches)

def main():
    quick_umls_matcher = QuickUMLS_Matcher()

    raw_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/data/combined_deidentified.jsonl"
    import json

    with open(raw_file, "r", encoding='utf-8') as in_H:
        for line in in_H:
            raw = json.loads(line.strip())
            print(f"encounterId {raw['encounterId']}")
            text = raw['pdf_text']


if __name__ == '__main__':
    main()