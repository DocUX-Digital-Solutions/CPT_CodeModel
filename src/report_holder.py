'''
{
  "encounter_number": "3097243",
  "payer": "Alignment Health Plan",
  "personId": "ed896501-c774-4240-bec0-1cf85f59614a",
  "encounterId": "640ffba1-8d95-4a09-a791-188a80e0f6a3",
  "operative_document_id": "1f7d6503-429a-40cf-98db-c05dba8705b6",
  "procedure_combinations": [
    {
      "cpt4Code": "29823",
      "modifierId1": "LT",
      "modifierId2": null,
      "modifierId3": null,
      "modifierId4": null,
      "diagnosisCodeId1": "M19.012"
    },
    {
      "cpt4Code": "29828",
      "modifierId1": "LT",
      "modifierId2": null,
      "modifierId3": null,
      "modifierId4": null,
      "diagnosisCodeId1": "M75.22"
    },
    {
      "cpt4Code": "29826",
      "modifierId1": "LT",
      "modifierId2": null,
      "modifierId3": null,
      "modifierId4": null,
      "diagnosisCodeId1": "M75.42"
    },
    {
      "cpt4Code": "29827",
      "modifierId1": "LT",
      "modifierId2": null,
      "modifierId3": null,
      "modifierId4": null,
      "diagnosisCodeId1": "M75.122"
    }
  ],
  "pdf_text":
'''
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass(frozen=True)
class ProcedureCombination:
    cpt4Code: str
    modifierId1: Optional[str]
    modifierId2: Optional[str]
    modifierId3: Optional[str]
    modifierId4: Optional[str]
    diagnosisCodeId1: str

@dataclass(frozen=True)
class ReportHolder:
    encounter_number: str
    payer: str
    personId: str
    encounterId: str
    operative_document_id: str
    procedure_combinations: List[ProcedureCombination]
    pdf_text: str

    @classmethod
    def from_dict(cls,
                  raw: Dict):
        loc = raw.copy()
        loc["procedure_combinations"] = [ProcedureCombination(**r) for r in loc["procedure_combinations"]]

        return cls(**loc)
