import spacy
import scispacy
from scispacy.linking import EntityLinker

from typing import Iterable, Tuple

class SciSpacyInterface:
    def __init__(self,
                 *,
                 model_name: str = "en_core_sci_scibert",
                 max_return: int = 3):
        self._model_name = model_name
        self.nlp = spacy.load(model_name)
        self.nlp.add_pipe("scispacy_linker",
                          config={"resolve_abbreviations": True, "linker_name": "umls"})

        self.linker = self.nlp.get_pipe("scispacy_linker")
        self._max_return = max_return

    def proc_string(self,
                    text: str,
                    *,
                    max_return: int = None) -> Iterable:
        if max_return is None:
            max_return = self._max_return

        doc = self.nlp(text)
        for ent in doc.ents:
            # print(f"\nEntity: {ent.text}")
            concepts = []
            for cui, score in ent._.kb_ents[:max_return]:
                concept = self.linker.kb.cui_to_entity[cui]
                concepts.append([score, concept])
            yield ent, concepts
                # print(f"  CUI: {cui} "
                #       f"| Score: {score:.3f} "
                #       f"| Type: {concept.type} "
                #       f"| Name: {concept.canonical_name}")

def main():
    from ml_util.cpt_holder import get_raw_code_table
    code_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/Consolidated_Code_List.txt"
    from UMLS.UMLS_ShortestPath import UMLSCache

    umls_cache = UMLSCache.load('/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/cache_2026AA')

    from UMLS.quick_umls import QuickUMLS_Matcher
    quick_umls_matcher = QuickUMLS_Matcher()

    raw_cpt = get_raw_code_table(code_file)
    code_inventory = raw_cpt.give_inventory(min_form_count_per_class=2,
                                            name="CPT Inventory",
                                            max_similarity=3)

    from umls_interface import SemGroups
    umls_groups = SemGroups()

    # interface = SciSpacyInterface(model_name="en_core_sci_lg")
    interface = None
    import re
    for m in code_inventory.members:
        if re.match(r"2[0-9]{4}$", m.label):
            text = m.representations[0]
            cpt_cui = umls_cache.code_to_cui.get(m.label)

            if interface is None:
                continue

            def give_group(gs):
                raw = [umls_groups.by_code[g].short_type for g in gs]
                return ';'.join(list(set(raw)))

            def show_match(m):
                score, entity = m
                path_to_cpt = umls_cache.shortest_path_between_cuis(cpt_cui, entity.concept_id)
                return f"{score:.3f} dist_to_cpt: {len(path_to_cpt)} {entity.canonical_name} {entity.concept_id} {give_group(entity.types)}"

            def show_entity(e):
                entries = '|'.join(
                    [show_match(m) for m in e[1]]
                )
                return f"term: {e[0]} ({e[0].start_char}-{e[0].end_char}) matches: {entries}"

            matches = [loc for loc in interface.proc_string(text)]

            def give_dep_inds(e) -> Tuple:
                t = e.root
                node_begin = []
                while True:
                    node_begin.insert(0, t.i)
                    if t.dep_ == 'ROOT':
                        break
                    else:
                        t = t.head
                return tuple(node_begin)

            by_nodes = sorted([(m, give_dep_inds(m[0])) for m in matches],
                              key=lambda x: (len(x[1]), x[1]))

            print(f"cpt: {m.label} text: {text}")
            for i, (m, path) in enumerate(by_nodes):
                print(f"{i}\t{show_entity(m)} path: {path}")

            pass


if __name__ == '__main__':
    main()