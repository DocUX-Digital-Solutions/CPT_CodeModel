import argparse
from ml_util.umls.graph_umls import UMLS_Tracker, init_cpt_worker, work_cpt
from ml_util.cpt_holder import get_raw_code_table
from ml_util.multi import multi_cpu_iter_global


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

    # umls_cache = UMLSCache.load(umls_cache_dir)

    umls_tracker = UMLS_Tracker(class_inventory=code_inventory, idf_method='double_normalization')

    # with cProfile.Profile() as pr:
    if True:
        # for m_ind, o in enumerate(multi_thread_cpu_iter(work_cpt, [list(code_inventory.members)],
        #                                                 constant_args=[umls_cache, args.descendant_depth],
    #                                                 num_per_cpu=1)):
        for m_ind, o in enumerate(multi_cpu_iter_global(work_cpt,
                                                        local_arg_list=[list(code_inventory.members)],
                                                        global_init_method=init_cpt_worker,
                                                        init_args=[umls_cache_dir, args.descendant_depth],
                                                        num_per_cpu=1,
                                                        max_workers=2)):
            # for m_ind, m in enumerate(code_inventory.members):
            # o = work_cpt(args.descendant_depth, m)
            if o is None:
                continue
            umls_tracker.add_entry(*o)
            if m_ind % 10 == 0:
                print(f"m_ind: {m_ind}")
                # stats = pstats.Stats(pr)
                # stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)  # Prints top 10 bottlenecks

    type_weights = umls_tracker.type_weights
    pass


if __name__ == '__main__':
    main()
