import random

from collections import Counter, defaultdict

from typing import Tuple

from ml_util.BM25_interface import BM25Index


def main():
    from ml_util.cpt_holder import get_raw_code_table
    code_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/Consolidated_Code_List.txt"
    from ml_util.umls.graph_umls import UMLSCache

    umls_cache = UMLSCache.load('/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/cache_2026AA')

    from ml_util.umls.quick_umls import QuickUMLS_Matcher
    quick_umls_matcher = QuickUMLS_Matcher()

    raw_cpt = get_raw_code_table(code_file)
    code_inventory = raw_cpt.give_inventory(min_form_count_per_class=2,
                                            name="CPT Inventory",
                                            max_similarity=3)

    from ml_util.umls.umls_sem_classes import SemGroups
    umls_groups = SemGroups()

    # interface = SciSpacyInterface(model_name="en_core_sci_lg")
    interface = None
    import re
    omit_hidden= [re.compile(r"\([^)]+\)"),
                  re.compile(r"\[[^]]+]")]
    skip_retired = True
    members = code_inventory.members[:]
    members = list(members)
    print(f"code members: {len(members)}")
    random.shuffle(members)
    for m in members:
        if re.match(r"^[0-9]{5}$", m.label):
            text = m.representations[0]
            cpt_cuis = umls_cache.code_to_cui.get(m.label, [])

            if interface is None:
                print(f"Start CPT: {m.label}")
                main = Counter()
                sub = Counter()
                rel_ratio = 2.0

                def add_list(l, c, *, norm_term = 1.0):
                    if skip_retired:
                        l = [i for i in l if not i.startswith('retired ')]
                    inc = 1.0/(float(len(l)) * norm_term)
                    for loc in l:
                        for o in omit_hidden:
                            loc = o.sub("", loc)
                        for t in set(BM25Index.tokenize(loc)):
                            c[t] += inc
                            pass

                all_sub = []
                feat_cnt = 0
                rel_level_tallies = Counter()
                for cpt_cui in cpt_cuis:
                    print(f"CPT: {m.label} -> {cpt_cui} -- {'|'.join(umls_cache.cui_to_terms[cpt_cui])}")
                    add_list(umls_cache.cui_to_terms[cpt_cui], main, norm_term=len(cpt_cuis))

                    use_cpt_cui, features, all_rel = umls_cache.get_features(cpt_cui)
                    if len(features):
                        feat_cnt += len(features)
                        # rel_cnt += sum([len(v) for v in all_rel.values()])
                        rel_level_tallies.update([v for rel in all_rel.values()
                                                  for v in rel.values()])
                        all_sub.append((use_cpt_cui, features, all_rel))

                for use_cpt_cui, features, all_rel in all_sub:
                    for f in sorted(features):
                        print(f"{(f[0], f[1])}\t{f[2]}\t{umls_cache.cui_to_stn[f[2]]}\t{'|'.join(umls_cache.cui_to_terms[f[2]])}")
                        add_list(umls_cache.cui_to_terms[f[2]], sub, norm_term=feat_cnt)
                        loc = all_rel.get(f[-1])
                        if loc:
                            by_level = defaultdict(set)
                            for k, v in loc.items():
                                by_level[v].add(k)

                            for level, for_level in by_level.items():
                                for r in for_level:
                                    print(f"\t{level}\t{r}\t{umls_cache.cui_to_stn[r]}\t{'|'.join(umls_cache.cui_to_terms[r])}")
                                    add_list(umls_cache.cui_to_terms[r], sub,
                                             norm_term=rel_level_tallies[level] * len(for_level) * rel_ratio * level)
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