from ml_util.umls.graph_umls import UMLSCache


def main():
    import argparse
    import json


    parser = argparse.ArgumentParser()
    parser.add_argument('--umls_cache_dir', type=str,
                        default='/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/cache_2026AA_kit_adamic_morePrune_undirected')
    args = parser.parse_args()

    umls_cache = UMLSCache.load(args.umls_cache_dir)

    def show_path(cpt0, cpt1):
        cui0 = umls_cache.code_to_cui[cpt0]
        cui1 = umls_cache.code_to_cui[cpt1]
        path = umls_cache.shortest_paths_between_cui_sets(cui0, cui1, cutoff=30)

        def show_cuis(all_c):
            if isinstance(all_c, str):
                all_c = [all_c]

            return "; ".join([f"{c} "
                              f"{umls_cache.cui_to_stn.get(c)} "
                              f"{','.join(umls_cache.sab_for_cui.get(c, []))} "
                              f"'{umls_cache.cui_to_preferred_term.get(c)}'"
                              for c in all_c])

        out = []
        for s, for_s in path.items():
            out.append(f"s: {show_cuis(s)}")
            for t, for_t in for_s.items():
                out.append(f"t: {show_cuis(t)}")
                out.append(f"score: {for_t[0]} len: {len(for_t[1])}")
                path_with_data = umls_cache.give_path_with_data(s, t, for_t[1])
                out.append(
                    umls_cache.show_path(path_with_data)
                )

        return "\n".join(out)

    # show_path('26460', '25290')
    show_path('27130', '27299')
    pass
    print(f"got here!")


main()