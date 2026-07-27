import argparse
from UMLS.UMLS_ShortestPath import UMLSCache
from ml_util.cpt_holder import get_raw_code_table, ClassInventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', type=str,
                        default="/Users/stevenfincke/PycharmProjects/CPT_CodeModel/Consolidated_Code_List.txt")
    parser.add_argument('--umls_cache_dir', type=str,
                        default='/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/cache_2026AA')
    args = parser.parse_args()

    code_file = args.code_file
    umls_cache_dir= args.umls_cache_dir


    raw_cpt = get_raw_code_table(code_file)
    code_inventory = raw_cpt.give_inventory(min_form_count_per_class=2,
                                            name="CPT Inventory",
                                            max_similarity=3)

    umls_cache = UMLSCache.load(umls_cache_dir)

    # Use InventoryTracker


if __name__ == '__main__':
    main()
