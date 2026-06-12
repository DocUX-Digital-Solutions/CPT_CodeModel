from typing import List, Dict, Tuple

from ml_util.classes import ClassInventory
from ml_util.docux_logger import give_logger

logger = give_logger()

class RawCPT:
    import re

    legal_form = re.compile(r"^[0-9]{5}$")

    skip_fields = ('Concept Id', 'Current Descriptor Effective Date', 'Test Name', 'Lab Name', 'Manufacturer Name',
                   'Spanish Consumer')
    '''
    Concept Id	CPT Code	Long	Medium	Short	Consumer	Spanish Consumer	Current Descriptor Effective Date	
    Test Name	Lab Name	Manufacturer Name
    '''
    header_begin = "Concept Id"
    code_field_name = 'CPT Code'
    display_fields: List[str] = ['Long', 'Consumer']
    field_sep = "\t"
    target_len = 5
    similarity_measure = 'common_init'

    def normalize_code(self,
                       code):
        return code

    @staticmethod
    def can_use_line(line: str) -> bool:
        return bool(len(line) > 0)

    def __init__(self,
                 code_file: str,
                 *,
                 required_init_strings: List[str] = None,
                 required_fields: List[str] = None,
                 digit_only: bool = False,
                 ):
        if isinstance(required_init_strings, list) and len(required_init_strings) < 1:
            required_init_strings = None
        self.by_code: Dict[str, Tuple[str]] = {}
        self.header_inds = []
        self.field_names: List[str] = []

        code_is_usable = \
            lambda code: ((required_init_strings is None
                           or sum([int(code.startswith(init_s)) for init_s in required_init_strings]) > 0)
                          and (digit_only is False or self.legal_form.match(code)))

        code_ind = None
        required_inds = None
        with open(code_file, "r", encoding='utf-8') as in_H:
            for line in in_H:
                line = line.strip()
                if len(self.header_inds) < 1:
                    if line.startswith(self.header_begin):
                        fields = line.strip().split(self.field_sep)
                        for ind, field in enumerate(fields):
                            if field not in self.skip_fields:
                                self.header_inds.append(ind)
                                self.field_names.append(field)
                        code_ind = self.field_names.index(self.code_field_name)
                        if required_fields:
                            required_inds = [self.field_names.index(n) for n in required_fields]
                else:
                    line = line.strip()
                    if not self.can_use_line(line):
                        continue
                    raw: List[str] = line.split(self.field_sep)
                    use_values = tuple([raw[i] if i < len(raw) else ''
                                        for i in self.header_inds])
                    if required_fields and min([len(use_values[ind]) for ind in required_inds]) < 1:
                        # logger.info(f"skip input line because it lacks required values: {line.strip()}")
                        continue
                    code = use_values[code_ind]
                    if code_is_usable(code):
                        self.by_code[self.normalize_code(code)] = use_values

        self.value_for_code_field = lambda code, field: (
            self.by_code[
                code
            ][
                self.field_names.index(field)
            ])
        pass

    @property
    def codes(self) -> List[str]:
        return sorted(list(self.by_code.keys()))

    # def give_variants_for_codes(self,
    #                             code: str) -> Tuple:
    #     return tuple([self.value_for_code_field(code, f)
    #                   for f in  self.display_fields])

    def give_inventory(self,
                       min_form_count_per_class: int,
                       max_similarity: int,
                       *,
                       name: str = 'CPT Inventory') -> ClassInventory:
        class_inventory = ClassInventory(name=name, max_similarity=max_similarity,
                                         similarity_measure=self.similarity_measure,
                                         strings_per_class=min_form_count_per_class)

        for code, fields in sorted(self.by_code.items()):
            ready_fields = sorted(list(set(
                [
                    fields[
                        self.field_names.index(n)
                    ]
                    for n in self.display_fields]
            )))
            if len(ready_fields) >= min_form_count_per_class:
                class_inventory.add_member(code, tuple(ready_fields))

        return class_inventory


class RawICD10(RawCPT):
    import re

    legal_form = re.compile(f"[A-Z][0-9]{2}"
                            f"("
                            f"[\.]?"
                            f"[0-9X]{1-3}"
                            f"[A-Z]?"
                            f")?")
    skip_fields = ('icd_version')
    display_fields = ['long_title']
    code_field_name = 'icd_code'
    header_begin = code_field_name
    field_sep = ","
    target_len = 7
    similarity_measure = 'class_3'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    @staticmethod
    def can_use_line(line: str) -> bool:
        return bool(',10,' in line)

    @staticmethod
    def normalize_code(code):
        code = code.replace('.', '')
        # code += ''.join([RawICD10.pad_char] * (RawICD10.target_len - len(code)))

        return code

task_class_map = {'CPT': RawCPT,
                  'ICD10': RawICD10}
supported_tasks = ('CPT', 'ICD10')


def get_raw_code_table(code_file: str,
                       *,
                       task_type: List[str] = supported_tasks[0],
                       required_fields: List[str] = None,
                       required_init_strings: List[str] = None,
                       ) -> RawCPT:
    if task_type not in task_class_map:
        raise ValueError(f"Unsupported task type: {task_type}")

    return task_class_map[task_type](code_file, required_fields=required_fields,
                                     required_init_strings=required_init_strings)
