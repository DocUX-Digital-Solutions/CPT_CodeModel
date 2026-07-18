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
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

from typing import List

csv.field_size_limit(sys.maxsize)

GRAPH_FILE = "graph.gpickle"
TERM_TO_CUIS_FILE = "term_to_cuis.pkl"
CUI_TO_PREFERRED_TERM_FILE = "cui_to_preferred_term.pkl"
CODE_TO_CUI_FILE = "code_to_cui.pkl"
SAB_FOR_CUI_FILE = "sab_for_cui.pkl"


class UMLSCache:
    """
    Wraps a UMLS-derived graph (CUIs as nodes, MRREL relationships as edges)
    plus term<->CUI lookup tables, and provides shortest-path and path-overlap
    queries over them.

    Construct via UMLSCache.build(...) the first time, and UMLSCache.load(...)
    on every subsequent run.
    """
    cpt_root_cuis = {'Current Procedural Terminology': 'C1138431'}
    snomed_sab = 'SNOMEDCT_US'

    def __init__(self, graph, term_to_cuis, cui_to_preferred_term, code_to_cui, sab_for_cui):
        self.graph = graph
        self.term_to_cuis = term_to_cuis
        self.cui_to_preferred_term = cui_to_preferred_term
        self.code_to_cui = code_to_cui
        self.sab_for_cui = sab_for_cui

    # ----------------------------------------------------------------
    # Construction: build from raw RRF files, or load a cached version
    # ----------------------------------------------------------------

    @classmethod
    def build(cls, mrconso_path, mrrel_path, cache_dir, languages=None, sabs=None, code_sabs=None):
        """
        Parse MRCONSO.RRF and MRREL.RRF, write the resulting graph and
        lookup tables to cache_dir, and return a ready-to-use UMLSCache.

        languages: iterable of LAT codes to keep (e.g. ["ENG"]). None = keep all.
        sabs: iterable of source vocab codes to keep (e.g. ["SNOMEDCT_US"]).
              None = keep all vocabularies.
        """
        term_to_cuis, cui_to_preferred_term, code_to_cui, sab_for_cui = cls._build_term_lookup(
            mrconso_path, languages=languages, sabs=sabs, code_sabs=code_sabs,
        )

        valid_cuis = set(cui_to_preferred_term.keys())
        graph = cls._build_graph(mrrel_path, sabs=sabs, valid_cuis=valid_cuis)

        cache = cls(graph, term_to_cuis, cui_to_preferred_term, code_to_cui, sab_for_cui)
        cache.save(cache_dir)
        return cache

    @classmethod
    def load(cls, cache_dir):
        """Load a previously-built cache from disk."""
        cache_dir = Path(cache_dir)
        with open(cache_dir / GRAPH_FILE, "rb") as f:
            graph = pickle.load(f)
        with open(cache_dir / TERM_TO_CUIS_FILE, "rb") as f:
            term_to_cuis = pickle.load(f)
        with open(cache_dir / CUI_TO_PREFERRED_TERM_FILE, "rb") as f:
            cui_to_preferred_term = pickle.load(f)
        with open(cache_dir / CODE_TO_CUI_FILE, "rb") as f:
            code_to_cui = pickle.load(f)
        with open(cache_dir / SAB_FOR_CUI_FILE, "rb")as f:
            sab_for_cui = pickle.load(f)
        return cls(graph, term_to_cuis, cui_to_preferred_term, code_to_cui, sab_for_cui)

    def save(self, cache_dir):
        """Write this cache's graph and lookup tables to cache_dir."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing cache to {cache_dir} ...")
        with open(cache_dir / GRAPH_FILE, "wb") as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_dir / TERM_TO_CUIS_FILE, "wb") as f:
            pickle.dump(self.term_to_cuis, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_dir / CUI_TO_PREFERRED_TERM_FILE, "wb") as f:
            pickle.dump(self.cui_to_preferred_term, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_dir / CODE_TO_CUI_FILE, "wb") as f:
            pickle.dump(self.code_to_cui, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_dir / SAB_FOR_CUI_FILE, "wb") as f:
            pickle.dump(self.sab_for_cui, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Cache written.")

    # ----------------------------------------------------------------
    # Internal parsing helpers (used by build())
    # ----------------------------------------------------------------

    @staticmethod
    def _build_term_lookup(mrconso_path, languages=None, sabs=None, code_sabs=None):
        """
        Parse MRCONSO.RRF into:
          - term_to_cuis: lowercase string -> set of CUIs
          - cui_to_preferred_term: CUI -> nicely-cased preferred string
        """
        languages = set(languages) if languages else None
        sabs = set(sabs) if sabs else None
        code_sabs = set(code_sabs) if code_sabs else None
        if code_sabs:
            if sabs is None or len(sabs.intersection(code_sabs)) < len(code_sabs):
                raise ValueError(f"All code_sab values must be included in sabs!")

        term_to_cuis = defaultdict(set)
        cui_to_preferred_term = {}
        code_to_cui = {}
        sab_for_cui = defaultdict(set)

        print(f"Parsing {mrconso_path} ...")
        with open(mrconso_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for i, row in enumerate(reader):
                if i % 500000 == 0 and i > 0:
                    print(f"  ...{i:,} MRCONSO rows processed")
                # MRCONSO columns (0-indexed):
                # 0 CUI, 1 LAT, 2 TS, 6 ISPREF, 11 SAB, 14 STR
                cui, lat, ts, ispref, sab, code, str_ = (
                    row[0], row[1], row[2], row[6], row[11], row[13], row[14]
                )
                if languages and lat not in languages:
                    continue
                if sabs and sab not in sabs:
                    continue
                if len(code) > 0 and sab in code_sabs:
                    code_to_cui[code] = cui
                sab_for_cui[cui].add(sab)

                term_to_cuis[str_.lower()].add(cui)

                if ts == "P" and ispref == "Y" and cui not in cui_to_preferred_term:
                    cui_to_preferred_term[cui] = str_

        sab_for_cui = {k: sorted(list(v)) for k, v in sab_for_cui.items()}

        # Fallback: any CUI without a captured preferred term just gets
        # whatever the first string we saw was, so downstream lookups don't
        # KeyError.
        for term, cuis in term_to_cuis.items():
            for cui in cuis:
                cui_to_preferred_term.setdefault(cui, term)

        print(f"Done. {len(term_to_cuis):,} unique terms, "
              f"{len(cui_to_preferred_term):,} CUIs with labels.")
        return dict(term_to_cuis), cui_to_preferred_term, code_to_cui, sab_for_cui

    @staticmethod
    def _build_graph(mrrel_path, sabs=None, valid_cuis=None):
        """
        Parse MRREL.RRF into an undirected networkx graph of CUIs.

        sabs: restrict to relationships asserted by these source vocabularies.
        valid_cuis: if given, only add edges where both CUIs are in this set
                    (keeps the graph aligned with a filtered term lookup).
        """
        sabs = set(sabs) if sabs else None
        G = nx.Graph()

        print(f"Parsing {mrrel_path} ...")
        with open(mrrel_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for i, row in enumerate(reader):
                if i % 1000000 == 0 and i > 0:
                    print(f"  ...{i:,} MRREL rows processed, "
                          f"{G.number_of_edges():,} edges so far")
                # MRREL columns (0-indexed):
                # 0 CUI1, 3 REL, 4 CUI2, 7 RELA, 10 SAB
                cui1, rel, cui2, rela, sab = row[0], row[3], row[4], row[7], row[10]

                if cui1 == cui2:
                    continue
                if sabs and sab not in sabs:
                    continue
                if valid_cuis and (cui1 not in valid_cuis or cui2 not in valid_cuis):
                    continue

                # Keep the most specific relationship label available on the edge.
                G.add_edge(cui1, cui2, rel=rel, rela=rela, sab=sab)

        print(f"Done. Graph has {G.number_of_nodes():,} nodes and "
              f"{G.number_of_edges():,} edges.")
        return G

    # ----------------------------------------------------------------
    # Term <-> CUI resolution
    # ----------------------------------------------------------------

    def resolve_term(self, term, verbose=True):
        """Return the set of CUIs matching a term string (case-insensitive, exact match)."""
        cuis = self.term_to_cuis.get(term.lower(), self.code_to_cui.get(term))
        if not cuis:
            raise ValueError(f"No CUI found for term: {term!r}")
        if isinstance(cuis, str):
            cuis = [cuis]
        if verbose and len(cuis) > 1:
            print(f"Note: '{term}' is ambiguous, matches {len(cuis)} CUIs: {sorted(cuis)}")
        return cuis

    def label(self, cui):
        """Preferred display term for a CUI, falling back to the CUI itself."""
        return self.cui_to_preferred_term.get(cui, cui)

    # ----------------------------------------------------------------
    # Shortest path queries
    # ----------------------------------------------------------------

    def shortest_path_between_cuis(self, cui1, cui2,
                                   *,
                                   method: str = 'dijkstra'):
        """Return the shortest CUI path between two CUIs, or None if unreachable."""
        if cui1 not in self.graph or cui2 not in self.graph:
            return None
        try:
            return nx.shortest_path(self.graph, cui1, cui2, method=method)
        except nx.NetworkXNoPath:
            return None

    def get_cpt_lowest_snomed_predecessor(self,
                                          cui: str,
                                          ):
        if not cui.startswith('C'):
            cui = self.resolve_term(cui)
        for c in self.shortest_path_between_cuis(cui, self.cpt_cui_root):
            if self.snomed_sab in self.sab_for_cui[c]:
                return c
        return None

    def describe_path_edges(self, cui_path):
        """
        Given a list of CUIs forming a path, return the relationship details
        (rel, rela, sab) stored on each edge connecting consecutive CUIs.

        Note: many UMLS relationships come as both a forward and inverse pair
        (e.g. PAR/CHD). Since the graph is undirected, if both directions
        exist between two nodes, only one edge is kept -- whichever was
        added during build(). If direction matters, query MRREL directly
        for that CUI pair.
        """
        edges = []
        for cui1, cui2 in zip(cui_path, cui_path[1:]):
            data = self.graph.get_edge_data(cui1, cui2) or {}
            edges.append({
                "from_cui": cui1,
                "to_cui": cui2,
                "rel": data.get("rel"),
                "rela": data.get("rela"),
                "sab": data.get("sab"),
            })
        return edges

    def get_edges_with_data(self,
                            cui):
        return [(p[1], self.graph.get_edge_data(*p))
                for p in self.graph.edges(cui)]

    def get_features(self,
                     cui: str):
        edges = self.get_edges_with_data(cui)
        if len(edges) == 1 and edges[0][1]['rela'] in {'has_clinician_form', 'has_consumer_friendly_form'}:
            cui = edges[0][0]
            edges = self.get_edges_with_data(cui)

        out = frozenset([f"{l[1]['rela']}:{l[0]}" for l in edges
                         if l[1]['rela'] not in {
                             'clinician_form_of', 'has_clinician_form', 'has_consumer_friendly_form',
                             'do_not_code_with', 'has_add_on_code', 'specialty_of', 'has_specialty', 'add_on_code_for',
                             'consumer_friendly_form_of', 'inverse_isa'}])

        return out

    def shortest_path_between_terms(self, term1, term2):
        """
        Resolve both terms to CUIs (trying every combination if either term
        is ambiguous), find the shortest CUI path for each combination, and
        return the overall shortest path translated back into readable terms.

        Returns None if no path exists, otherwise a dict with the CUI path,
        the term path, per-hop relationship details, and which CUIs were
        used to anchor each term.
        """
        cuis1 = self.resolve_term(term1)
        cuis2 = self.resolve_term(term2)

        best_path = None
        best_pair = None
        for c1 in cuis1:
            for c2 in cuis2:
                path = self.shortest_path_between_cuis(c1, c2)
                if path and (best_path is None or len(path) < len(best_path)):
                    best_path = path
                    best_pair = (c1, c2)

        if not best_path:
            return None

        return {
            "cui_path": best_path,
            "term_path": [self.label(cui) for cui in best_path],
            "edges": self.describe_path_edges(best_path),
            "start_cui": best_pair[0],
            "end_cui": best_pair[1],
            "hops": len(best_path) - 1,
        }

    # ----------------------------------------------------------------
    # Path overlap
    # ----------------------------------------------------------------

    @staticmethod
    def path_overlap(path1, path2):
        """
        Compare two CUI paths (each a list of CUIs) and report where they
        overlap.

        Returns a dict with:
          - shared_nodes: CUIs present in both paths, in path1's order
          - shared_edges: (cui_a, cui_b) pairs traversed as a direct hop
                           in both paths, regardless of direction
          - node_positions: {cui: (index_in_path1, index_in_path2)}
          - has_overlap: bool
        """
        set2 = set(path2)
        shared_nodes = [cui for cui in path1 if cui in set2]

        edges1 = {frozenset((a, b)) for a, b in zip(path1, path1[1:])}
        edges2 = {frozenset((a, b)) for a, b in zip(path2, path2[1:])}
        shared_edges = [tuple(sorted(e)) for e in (edges1 & edges2)]

        node_positions = {
            cui: (path1.index(cui), path2.index(cui)) for cui in shared_nodes
        }

        return {
            "shared_nodes": shared_nodes,
            "shared_edges": shared_edges,
            "node_positions": node_positions,
            "has_overlap": bool(shared_nodes),
        }

    def describe_path_overlap(self, path1, path2):
        """Human-readable version of path_overlap: nodes/edges as preferred terms."""
        overlap = self.path_overlap(path1, path2)
        return {
            "shared_nodes": [(cui, self.label(cui)) for cui in overlap["shared_nodes"]],
            "shared_edges": [
                (a, self.label(a), b, self.label(b)) for a, b in overlap["shared_edges"]
            ],
            "node_positions": overlap["node_positions"],
            "has_overlap": overlap["has_overlap"],
        }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_path_result(result):
    print(f"\nShortest path ({result['hops']} hop{'s' if result['hops'] != 1 else ''}):")
    print("  " + "  ->  ".join(result["term_path"]))
    print("\nCUIs:")
    print("  " + "  ->  ".join(result["cui_path"]))
    print("\nRelationship detail per hop:")
    for e in result["edges"]:
        print(f"  {e['from_cui']} --[rel={e['rel']}, rela={e['rela']}, sab={e['sab']}]--> {e['to_cui']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Parse raw RRF files and write a cache to disk.")
    build_p.add_argument("--mrconso", required=True, help="Path to MRCONSO.RRF")
    build_p.add_argument("--mrrel", required=True, help="Path to MRREL.RRF")
    build_p.add_argument("--cache-dir", required=True, help="Directory to write the cache into")
    build_p.add_argument("--languages", nargs="*", default=["ENG"],
                          help="LAT codes to keep (default: ENG). Pass nothing to keep all.")
    build_p.add_argument("--sabs", nargs="*", default=None,
                          help="Source vocabularies to keep (e.g. SNOMEDCT_US RXNORM). "
                               "Default: keep all vocabularies.")
    build_p.add_argument("--code_sabs", nargs="*", default=None,
                         help="Source vocabularies to keep (e.g. SNOMEDCT_US RXNORM) to keep codes. "
                              "Default: store no codes.")

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
        UMLSCache.build(args.mrconso, args.mrrel, args.cache_dir,
                         languages=languages, sabs=args.sabs, code_sabs=args.code_sabs)
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
