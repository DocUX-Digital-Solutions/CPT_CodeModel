#!/usr/bin/env python3
"""
umls_shortest_path.py

A UMLSCache class that builds a graph + term lookup from UMLS Metathesaurus
RRF files (MRCONSO.RRF, MRREL.RRF), caches them to disk, and lets you query
shortest paths between terms or CUIs -- including relationship detail per
hop and overlap analysis between two paths.

Usage
-----
1) Build the cache once (this parses the raw RRF files and can take a
   while depending on how much of UMLS you keep):

    python umls_shortest_path.py build \
        --mrconso /path/to/MRCONSO.RRF \
        --mrrel /path/to/MRREL.RRF \
        --cache-dir ./umls_cache \
        --languages ENG \
        --sabs SNOMEDCT_US RXNORM

   Omit --sabs to keep all source vocabularies (much bigger graph).

2) Query as many times as you want -- this just loads the cache:

    python umls_shortest_path.py path \
        --cache-dir ./umls_cache \
        --term1 "myocardial infarction" \
        --term2 "aspirin"

    python umls_shortest_path.py overlap \
        --cache-dir ./umls_cache \
        --terms "aspirin" "stroke" "warfarin" "bleeding"

As a library
------------
    from umls_shortest_path import UMLSCache

    # Build once (writes cache to disk)
    cache = UMLSCache.build(
        mrconso_path="/path/to/MRCONSO.RRF",
        mrrel_path="/path/to/MRREL.RRF",
        cache_dir="./umls_cache",
        languages=["ENG"],
        sabs=["SNOMEDCT_US", "RXNORM"],
    )

    # Or load an existing cache (fast, this is what you'll normally do)
    cache = UMLSCache.load("./umls_cache")

    result = cache.shortest_path_between_terms("myocardial infarction", "aspirin")
    print(result["term_path"])
    print(result["edges"])  # rel/rela/sab per hop

    path1 = cache.shortest_path_between_terms("aspirin", "stroke")["cui_path"]
    path2 = cache.shortest_path_between_terms("warfarin", "bleeding")["cui_path"]
    overlap = cache.describe_path_overlap(path1, path2)
"""

import argparse

from ml_util.network_interface import GraphHolder
from ml_util.umls.graph_umls import UMLSCache, _print_path_result



# TERM_TO_CUIS_FILE = "term_to_cuis.pkl"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Parse raw RRF files and write a cache to disk.")
    build_p.add_argument("--mrconso", required=True, help="Path to MRCONSO.RRF")
    build_p.add_argument("--mrrel", required=True, help="Path to MRREL.RRF")
    build_p.add_argument("--mrsty", required=True, help="Path to MRSTY.RRF")
    build_p.add_argument("--cache-dir", required=True, help="Directory to write the cache into")
    build_p.add_argument("--languages", nargs="*", default=["ENG"],
                          help="LAT codes to keep (default: ENG). Pass nothing to keep all.")
    build_p.add_argument("--sabs", nargs="*", default=None,
                          help="Source vocabularies to keep (e.g. SNOMEDCT_US RXNORM). "
                               "Default: keep all vocabularies.")
    build_p.add_argument('--weighting', type=str, choices=GraphHolder.supported_weighting_schemes)
    build_p.add_argument('--centrality_part_cutoff', type=float)
    build_p.add_argument('--prune_skip_sty', type=str, nargs='*')
    build_p.add_argument('--prune_skip_class_sty', type=str, nargs='*')

    path_p = sub.add_parser("path", help="Find the shortest path using an existing cache.")
    path_p.add_argument("--cache-dir", required=True, help="Directory containing the built cache")
    path_p.add_argument("--term1", help="First term (mutually exclusive with --cui1)")
    path_p.add_argument("--term2", help="Second term (mutually exclusive with --cui2)")
    path_p.add_argument("--cui1", help="First CUI (mutually exclusive with --term1)")
    path_p.add_argument("--cui2", help="Second CUI (mutually exclusive with --term2)")

    overlap_p = sub.add_parser(
        "overlap",
        help="Find two shortest paths (A->B and C->D) and report where they overlap."
    )
    overlap_p.add_argument("--cache-dir", required=True, help="Directory containing the built cache")
    overlap_p.add_argument("--terms", nargs=4, metavar=("TERM_A", "TERM_B", "TERM_C", "TERM_D"),
                            help="Four terms: path1 = TERM_A -> TERM_B, path2 = TERM_C -> TERM_D")
    overlap_p.add_argument("--cuis", nargs=4, metavar=("CUI_A", "CUI_B", "CUI_C", "CUI_D"),
                            help="Four CUIs: path1 = CUI_A -> CUI_B, path2 = CUI_C -> CUI_D")

    args = parser.parse_args()

    if args.command == "build":
        languages = args.languages if args.languages else None
        if args.centrality_part_cutoff is not None:
            from ml_util.umls.umls_sem_classes import SemGroups
            sem_group_table = SemGroups()
            prune_skip_sty = []
            if args.prune_skip_class_sty and len(args.prune_skip_class_sty) > 0:
                for c in args.prune_skip_class_sty:
                    prune_skip_sty.extend(
                        [e.code for e in sem_group_table.by_short[c]]
                    )
            if args.prune_skip_sty and len(args.prune_skip_sty) > 0:
                prune_skip_sty.extend(args.prune_skip_sty)
            prune_skip_sty = sorted(list(set(prune_skip_sty)))
            assert all([p in sem_group_table.by_code for p in prune_skip_sty])
        else:
            prune_skip_sty = None

        UMLSCache.build(args.mrconso, args.mrrel, args.mrsty, args.cache_dir,
                        languages=languages, sabs=args.sabs, weighting=args.weighting,
                        centrality_part_cutoff=args.centrality_part_cutoff, prune_skip_sty=prune_skip_sty)
        return

    if args.command == "path":
        cache = UMLSCache.load(args.cache_dir)

        using_terms = args.term1 or args.term2
        using_cuis = args.cui1 or args.cui2
        if using_terms and using_cuis:
            parser.error("Use either --term1/--term2 or --cui1/--cui2, not both.")
        if using_terms and not (args.term1 and args.term2):
            parser.error("Both --term1 and --term2 are required.")
        if using_cuis and not (args.cui1 and args.cui2):
            parser.error("Both --cui1 and --cui2 are required.")

        if using_terms:
            result = cache.shortest_path_between_terms(args.term1, args.term2)
            if not result:
                print(f"No path found between '{args.term1}' and '{args.term2}'.")
                return
            _print_path_result(result)
        else:
            cui_path = cache.shortest_path_between_cuis(args.cui1, args.cui2)
            if not cui_path:
                print(f"No path found between {args.cui1} and {args.cui2}.")
                return
            result = {
                "cui_path": cui_path,
                "term_path": [cache.label(c) for c in cui_path],
                "edges": cache.describe_path_edges(cui_path),
                "hops": len(cui_path) - 1,
            }
            _print_path_result(result)
        return

    if args.command == "overlap":
        cache = UMLSCache.load(args.cache_dir)

        if not (args.terms or args.cuis) or (args.terms and args.cuis):
            parser.error("Provide exactly one of --terms A B C D or --cuis A B C D.")

        if args.terms:
            a, b, c, d = args.terms
            result1 = cache.shortest_path_between_terms(a, b)
            result2 = cache.shortest_path_between_terms(c, d)
            if not result1:
                print(f"No path found between '{a}' and '{b}'.")
                return
            if not result2:
                print(f"No path found between '{c}' and '{d}'.")
                return
            path1, path2 = result1["cui_path"], result2["cui_path"]
            print(f"Path 1 ({a} -> {b}): " + "  ->  ".join(result1["term_path"]))
            print(f"Path 2 ({c} -> {d}): " + "  ->  ".join(result2["term_path"]))
        else:
            a, b, c, d = args.cuis
            path1 = cache.shortest_path_between_cuis(a, b)
            path2 = cache.shortest_path_between_cuis(c, d)
            if not path1:
                print(f"No path found between {a} and {b}.")
                return
            if not path2:
                print(f"No path found between {c} and {d}.")
                return
            print(f"Path 1 ({a} -> {b}): " + "  ->  ".join(path1))
            print(f"Path 2 ({c} -> {d}): " + "  ->  ".join(path2))

        overlap = cache.describe_path_overlap(path1, path2)

        print()
        if not overlap["has_overlap"]:
            print("No overlapping concepts between the two paths.")
            return

        print(f"Shared concepts ({len(overlap['shared_nodes'])}):")
        for cui, term in overlap["shared_nodes"]:
            i1, i2 = overlap["node_positions"][cui]
            print(f"  {term} ({cui})  [position {i1} in path1, position {i2} in path2]")

        if overlap["shared_edges"]:
            print(f"\nShared direct hops ({len(overlap['shared_edges'])}):")
            for cui_a, term_a, cui_b, term_b in overlap["shared_edges"]:
                print(f"  {term_a} ({cui_a})  <->  {term_b} ({cui_b})")
        else:
            print("\nNo shared direct hops (overlap is at shared concepts only, "
                  "not shared edges).")


if __name__ == "__main__":
    main()
