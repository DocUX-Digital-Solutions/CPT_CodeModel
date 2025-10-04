from typing import List, Dict, Tuple

from ml_util.classes import ClassInventory
from ml_util.docux_logger import give_logger

logger = give_logger()

class RawCPT:
    skip_fields = ('Concept Id', 'Current Descriptor Effective Date', 'Test Name', 'Lab Name', 'Manufacturer Name',
                   'Spanish Consumer')
    '''
    Concept Id	CPT Code	Long	Medium	Short	Consumer	Spanish Consumer	Current Descriptor Effective Date	
    Test Name	Lab Name	Manufacturer Name
    '''
    display_fields: List[str] = ['Long', 'Consumer']
    def __init__(self,
                 code_file: str,
                 *,
                 required_init_strings: List[str] = None,
                 required_fields: List[str] = None
                 ):
        self.by_cpt: Dict[str, Tuple[str]] = {}
        self.header_inds = []
        self.field_names: List[str] = []

        cpt_is_usable = \
            lambda cpt: (required_init_strings is None
                         or sum([int(cpt.startswith(init_s)) for init_s in required_init_strings]) > 0)

        cpt_ind = None
        required_inds = None
        with open(code_file, "r", encoding='utf-8') as in_H:
            for line in in_H:
                line = line.strip()
                if len(self.header_inds) < 1:
                    if line.startswith("Concept Id"):
                        fields = line.strip().split("\t")
                        for ind, field in enumerate(fields):
                            if field not in self.skip_fields:
                                self.header_inds.append(ind)
                                self.field_names.append(field)
                        cpt_ind = self.field_names.index('CPT Code')
                        if required_fields:
                            required_inds = [self.field_names.index(n) for n in required_fields]
                else:
                    line = line.strip()
                    raw: List[str] = line.split("\t")
                    use_values = tuple([raw[i] if i < len(raw) else ''
                                        for i in self.header_inds])
                    if required_fields and min([len(use_values[ind]) for ind in required_inds]) < 1:
                        # logger.info(f"skip input line because it lacks required values: {line.strip()}")
                        continue
                    cpt = use_values[cpt_ind]
                    if cpt_is_usable(cpt):
                        self.by_cpt[cpt] = use_values

        self.value_for_cpt_field = lambda cpt, field: (
            self.by_cpt[
                cpt
            ][
                self.field_names.index(field)
            ])
        pass

    def give_variants_for_cpt(self,
                              cpt_code: str) -> Tuple:
        return tuple([self.value_for_cpt_field(cpt_code, f)
                      for f in  self.display_fields])

    def give_inventory(self,
                       min_form_count_per_class: int) -> ClassInventory:
        class_inventory = ClassInventory(name='CPT Inventory')

        for cpt, fields in sorted(self.by_cpt.items()):
            ready_fields = sorted(list(set(
                [
                    fields[
                        self.field_names.index(n)
                    ]
                    for n in self.display_fields]
            )))
            if len(ready_fields) >= min_form_count_per_class:
                class_inventory.add_member(cpt, tuple(ready_fields))

        return class_inventory
