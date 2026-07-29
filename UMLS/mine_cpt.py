import argparse
from UMLS.UMLS_ShortestPath import UMLSCache, UMLS_Tracker
from ml_util.cpt_holder import get_raw_code_table, ClassInventory
from ml_util.multi import multi_thread_cpu_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', type=str,
                        default="/Users/stevenfincke/PycharmProjects/CPT_CodeModel/Consolidated_Code_List.txt")
    parser.add_argument('--umls_cache_dir', type=str,
                        default='/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/cache_2026AA_kit')
    parser.add_argument('--descendant_depth', type=int, default=1)
    args = parser.parse_args()

    code_file = args.code_file
    umls_cache_dir= args.umls_cache_dir


    raw_cpt = get_raw_code_table(code_file)
    code_inventory = raw_cpt.give_inventory(min_form_count_per_class=2,
                                            name="CPT Inventory",
                                            max_similarity=3)

    umls_cache = UMLSCache.load(umls_cache_dir)

    umls_tracker = UMLS_Tracker(class_inventory=code_inventory, idf_method='double_normalization')

    # # Use InventoryTracker
    # for member in code_inventory.members:
    #     cpt_cuis = umls_cache.code_to_cui.get(member.label)
    #     if not cpt_cuis:
    #         print(f"Missing: cpt: {member.label}")
    #         continue
    #     counts = defaultdict(list)
    #     for cpt_cui in cpt_cuis:
    #         _, immediate_feats, distant_feats = umls_cache.get_features(cpt_cui, descendant_depth=1)
    #         for f in immediate_feats:
    #             counts[f[2]].append(1.0)
    #         for v in distant_feats.values():
    #             for d, l in v.items():
    #                 counts[d].append(1.0 / (2.0 + l))
    #     umls_tracker.add_entry(member.label_ind, {k: max(v) for k, v in counts.items()})

    multi_thread_cpu_map(work_cpt, [[m for m in code_inventory.members if m.label.startswith('2')]],
                         constant_args=[umls_cache, umls_tracker, args.descendant_depth],
                         num_per_cpu=1)

    type_weights = umls_tracker.type_weights
    pass

from collections import defaultdict
def work_cpt(umls_cache, umls_tracker, descendant_depth, member):
    cpt_cuis = umls_cache.code_to_cui.get(member.label)
    if not cpt_cuis:
        print(f"Missing: cpt: {member.label}")
        return
    counts = defaultdict(list)
    for cpt_cui in cpt_cuis:
        _, immediate_feats, distant_feats = umls_cache.get_features(cpt_cui, descendant_depth=descendant_depth)
        for f in immediate_feats:
            counts[f[2]].append(1.0)
        for v in distant_feats.values():
            for d, l in v.items():
                counts[d].append(1.0 / (2.0 + l))
    umls_tracker.add_entry(member.label_ind, {k: max(v) for k, v in counts.items()})


if __name__ == '__main__':
    main()
