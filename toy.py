from dataclasses import dataclass
from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder
from typing import List, Dict, Optional
from collections import Counter


@dataclass(frozen=True)
class CPT_entry:
    Concept_Id : str
    CPT_Code: str
    Long: str
    Medium: str # all-caps, not normalized
    Short: str # all-caps, not normalized
    Consumer: Optional[str]
    Spanish_Consumer: Optional[str]
    Current_Descriptor_Effective_Date: Optional[str]
    Test_Name: Optional[str]
    Lab_Name: Optional[str]
    Manufacturer_Name: Optional[str]

    _first_field = 'Concept Id'
    _key_field = 'CPT Code'

    def __contains__(self, item):
        if isinstance(item, str):
            return any([item in f for f in (self.Long, self.Consumer, self.Medium, self.Short)])

        raise NotImplementedError

    @property
    def key(self) -> str:
        return self.CPT_Code


@dataclass(frozen=True)
class CPT_table:
    _field_names: List[str]
    codes: Dict[str, CPT_entry]

    _first_field = 'Concept Id'
    _key_field = 'CPT Code'

    def __getitem__(self, item):
        return self.codes[item]

    def __contains__(self, item):
        return item in self.codes

    @classmethod
    def create(cls,
               cpt_table = "Consolidated_Code_List.txt"):

        table: Dict[str, CPT_entry] = {}

        field_names = None
        with (open(cpt_table, "r", encoding='utf-8') as in_H):
            for line in in_H:
                line.strip("\n")
                if field_names is None:
                    if line.startswith(CPT_entry._first_field):
                        field_names = [f.strip().replace(" ", "_") for  f in line.split("\t")]
                else:
                    parts = [f if len(f) > 0 else None for f in
                             [s.strip() for s in line.split("\t")]]
                    assert len(parts) == len(field_names)
                    entry = CPT_entry(**{k: v for k, v in zip(field_names, parts)})
                    assert entry.key not in table
                    table[entry.key] = entry

        return cls(field_names, table)


def get_cpt_tally(code_tally_csv: str = "CPT_Code_Distribution_NO_PHI_2.txt"):
    cpt_tally: Dict[str, int] = Counter()
    with (open(code_tally_csv, "r", encoding='utf-8') as in_H):
        field_names = None
        for line in in_H:
            line = line.strip()
            if len(line) < 1:
                continue
            parts = line.split("\t\t")
            if field_names is None:
                assert parts[0] == "CPT4 Code"
                field_names = parts[:]
            else:
                cpt_code = parts[0]
                if len(cpt_code) == 5:
                    cpt_tally[cpt_code] += int(float(parts[1]))
    cpt_tally = dict(cpt_tally)

    return cpt_tally

cpt_tally = get_cpt_tally()

cpt_reference_table = CPT_table.create()
assert all([k in cpt_reference_table
            for k in cpt_tally.keys()])

model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
sth = SentenceTransformerHolder.create(model_name=model_name)


long_texts = {}
for code, cnt in sorted([(k, v) for k, v in cpt_tally.items()], key=lambda x: (-x[1], x[0])):
    long_texts[code] = cpt_reference_table[code].Long
    print(f"code: {code}\tcnt: {cnt}\t{long_texts[code]}")

use_codes = sorted(long_texts.keys())
use_texts = [long_texts[c] for c in use_codes]
use_encodings = sth.encode(use_texts)
pair_inds, pair_dist = sth.single_pool_similarity(use_encodings)

for rank, (pi, d) in enumerate(zip(pair_inds, pair_dist)):
    print(f"{rank}\t{d}\n{use_codes[pi[0]]}\t{use_texts[pi[0]]}\n{use_codes[pi[1]]}\t{use_texts[pi[1]]}")

print(f"do 298XX")

use_codes = sorted([c for c in cpt_reference_table.codes.keys() if c.startswith('298')])
use_texts = [cpt_reference_table[c].Long for c in use_codes]
use_encodings = sth.encode(use_texts)
pair_inds, pair_dist = sth.single_pool_similarity(use_encodings)

for rank, (pi, d) in enumerate(zip(pair_inds, pair_dist)):
    print(f"{rank}\t{d}\n{use_codes[pi[0]]}\t{use_texts[pi[0]]}\n{use_codes[pi[1]]}\t{use_texts[pi[1]]}")



