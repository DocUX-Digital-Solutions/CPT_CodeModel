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
import networkit as nk

from typing import List, Dict, FrozenSet, Tuple, Set, Iterable

from ml_util.network_interface import GraphHolder
from scripts.LSI_complete import InventoryTracker

csv.field_size_limit(sys.maxsize)

GRAPH_FILE = "graph.gpickle"
# TERM_TO_CUIS_FILE = "term_to_cuis.pkl"
CUI_TO_TERMS_FILE = "cui_to_terms.pkl"
CUI_TO_PREFERRED_TERM_FILE = "cui_to_preferred_term.pkl"
CODE_TO_CUI_FILE = "code_to_cui.pkl"
SAB_FOR_CUI_FILE = "sab_for_cui.pkl"
CUI_TO_STN_FILE = "cui_to_stn.pkl"

from dataclasses import dataclass

@dataclass(frozen=True)
class DataRels:
    rel: None | str
    rela: None | str

    def as_dict(self) -> Dict:
        return {'rel': self.rel,
                'rela': self.rela}


@dataclass(frozen=True)
class DataForEdge:
    rel: str
    rela: str
    sab: str


@dataclass(frozen=True)
class EdgeWithData:
    first_cui: str
    second_cui: str
    all_data: FrozenSet[DataForEdge]

    def by_rel(self) -> Dict[str, List[DataForEdge]]:
        out = defaultdict(list)
        for e in self.all_data:
            out[e.rel].append(e)

        return dict(out)

    def has_rel(self,
                rels: Set):
        assert isinstance(rels, set)
        return any([d.rel in rels for d in self.all_data])

    def has_rela(self,
                 relas: Set):
        assert isinstance(relas, set)
        return any([d.rela in relas for d in self.all_data])

    def data_with_rel_and_rela(self,
                               rels: None | Set,
                               relas: None | Set):
        all_r = [rels, relas]
        assert all([r is None or isinstance(r, set) for r in all_r])
        assert sum([r is not None for r in all_r]) > 0

        return [d for d in self.all_data
                if all([s is None or v in s
                        for v, s in zip((d.rel, d.rela), all_r)])]

    def has_rel_and_rela(self,
                         rels: None | Set,
                         relas: None | Set) -> bool:
        return len(self.data_with_rel_and_rela(rels, relas)) > 0

    def matches_data_rel(self,
                         data_rel: DataRels | None):
        if data_rel is None:
            return True
        as_set = lambda x: None if x is None else {x}
        return len(self.data_with_rel_and_rela(as_set(data_rel.rel), as_set(data_rel.rela))) > 0

    @classmethod
    def create(cls,
               first_cui: str,
               second_cui: str,
               all_edge_data: List,
               ):
        if 0 in all_edge_data:
            all_data = frozenset([DataForEdge(**v)
                                  for k, v in
                                  sorted(all_edge_data.items())
                                  ])
        else:
            all_data = frozenset([DataForEdge(**ed) for ed in all_edge_data])

        return cls(first_cui, second_cui, all_data)


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

    acceptable_specs = {'PAR': (DataRels('PAR', 'inverse_isa'), DataRels('PAR', 'inverse_isa')),
                        'CHD': (DataRels('CHD', 'isa'), DataRels('CHD', 'isa')),
                        'RB': (DataRels('RB', None), DataRels('PAR', 'inverse_isa')),
                        'RN': (DataRels('RN', None), DataRels('CHD', 'isa')),
                        'RO': (DataRels('RO', None), DataRels('PAR', 'inverse_isa'))
                        }

    def __init__(self,
                 graph_holder: GraphHolder,
                 cui_to_terms,
                 cui_to_preferred_term,
                 code_to_cui,
                 sab_for_cui,
                 cui_to_stn):
        self.graph_holder = graph_holder
        self._member_list = self.graph_holder._node_names.member_list
        self._member_lookup = self.graph_holder._node_names.member_dict
        self.cui_to_terms = cui_to_terms
        self.cui_to_preferred_term = cui_to_preferred_term
        self.code_to_cui = code_to_cui
        self.cui_to_code = {v: k
                            for k, all_v in self.code_to_cui.items()
                            for v in all_v}
        self.sab_for_cui = sab_for_cui
        self.cui_to_stn = cui_to_stn

    # ----------------------------------------------------------------
    # Construction: build from raw RRF files, or load a cached version
    # ----------------------------------------------------------------

    @classmethod
    def build(cls, mrconso_path, mrrel_path, mrsty_path, cache_dir, languages=None, sabs=None, weighting=None,
              centrality_part_cutoff=None, prune_skip_sty=None):
        """
        Parse MRCONSO.RRF and MRREL.RRF, write the resulting graph and
        lookup tables to cache_dir, and return a ready-to-use UMLSCache.

        languages: iterable of LAT codes to keep (e.g. ["ENG"]). None = keep all.
        sabs: iterable of source vocab codes to keep (e.g. ["SNOMEDCT_US"]).
              None = keep all vocabularies.
        """
        if weighting:
            assert weighting in GraphHolder.supported_weighting_schemes

        term_to_cuis, cui_to_preferred_term, code_to_cui, sab_for_cui = cls._build_term_lookup(
            mrconso_path, languages=languages, sabs=sabs,
        )

        valid_cuis = set(cui_to_preferred_term.keys())

        cui_to_stn, cui_to_sty = cls._build_cui_to_stn(mrsty_path, valid_cuis=valid_cuis)

        graph = cls._build_graph_holder(mrrel_path, sabs=sabs, valid_cuis=valid_cuis, weighting=weighting,
                                        centrality_part_cutoff=centrality_part_cutoff, cui_to_sty=cui_to_sty,
                                        prune_skip_sty=prune_skip_sty)

        def proc_code_cuis(all_cuis) -> List[int]:
            if len(all_cuis) <= 1:
                return all_cuis

            neighbors = {c: graph.give_neighbors(c) for c in all_cuis}
            use_cuis = []
            covered_neighbors = set([])
            for k, v in sorted(neighbors.items(), key=lambda x: (-len(x[1]), x[0])):
                loc = set([k] + v)
                if len(covered_neighbors.intersection(loc)) < len(loc):
                    covered_neighbors.update(loc)
                    use_cuis.append(k)

            return use_cuis

        code_to_cui = {k: proc_code_cuis(v) for k, v in code_to_cui.items()}


        cache = cls(graph, term_to_cuis, cui_to_preferred_term, code_to_cui, sab_for_cui, cui_to_stn)
        cache.save(cache_dir)
        return cache

    @classmethod
    def load(cls, cache_dir):
        """Load a previously-built cache from disk."""
        cache_dir = Path(cache_dir)
        with open(cache_dir / GRAPH_FILE, "rb") as f:
            graph = pickle.load(f)
        graph.post_load(cache_dir / GRAPH_FILE)
        with open(cache_dir / CUI_TO_TERMS_FILE, "rb") as f:
            cui_to_terms = pickle.load(f)
        with open(cache_dir / CUI_TO_PREFERRED_TERM_FILE, "rb") as f:
            cui_to_preferred_term = pickle.load(f)
        with open(cache_dir / CODE_TO_CUI_FILE, "rb") as f:
            code_to_cui = pickle.load(f)
        with open(cache_dir / SAB_FOR_CUI_FILE, "rb")as f:
            sab_for_cui = pickle.load(f)
        with open(cache_dir / CUI_TO_STN_FILE, "rb")as f:
            cui_to_stn = pickle.load(f)
        return cls(graph, cui_to_terms, cui_to_preferred_term, code_to_cui, sab_for_cui, cui_to_stn)

    def save(self, cache_dir):
        """Write this cache's graph and lookup tables to cache_dir."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing cache to {cache_dir} ...")
        self.graph_holder.prepickle(cache_dir / GRAPH_FILE)
        with open(cache_dir / GRAPH_FILE, "wb") as f:
            pickle.dump(self.graph_holder, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_dir / CUI_TO_TERMS_FILE, "wb") as f:
            pickle.dump(self.cui_to_terms, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_dir / CUI_TO_PREFERRED_TERM_FILE, "wb") as f:
            pickle.dump(self.cui_to_preferred_term, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_dir / CODE_TO_CUI_FILE, "wb") as f:
            pickle.dump(self.code_to_cui, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_dir / SAB_FOR_CUI_FILE, "wb") as f:
            pickle.dump(self.sab_for_cui, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_dir / CUI_TO_STN_FILE, "wb") as f:
            pickle.dump(self.cui_to_stn, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Cache written.")

    # ----------------------------------------------------------------
    # Internal parsing helpers (used by build())
    # ----------------------------------------------------------------

    @staticmethod
    def _build_cui_to_stn(mrsty_path, valid_cuis=None) -> Tuple[Dict, Dict]:
        cui_to_stn = {}
        cui_to_sty = {}
        print(f"Parsing {mrsty_path} ...")
        with open(mrsty_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for i, row in enumerate(reader):
                if i % 500000 == 0 and i > 0:
                    print(f"  ...{i:,} MRSTY rows processed")
                cui, sty, stn  = (
                    row[0], row[1], row[2],
                )
                if valid_cuis and cui not in valid_cuis:
                    continue
                cui_to_stn[cui] = stn
                cui_to_sty[cui] = sty

        return cui_to_stn, cui_to_sty

    @staticmethod
    def _build_term_lookup(mrconso_path, languages=None, sabs=None):
        """
        Parse MRCONSO.RRF into:
          - term_to_cuis: lowercase string -> set of CUIs
          - cui_to_preferred_term: CUI -> nicely-cased preferred string
        """
        languages = set(languages) if languages else None
        sabs = set(sabs) if sabs else None

        cui_to_terms = defaultdict(set)
        cui_to_preferred_term = {}
        code_to_cui = defaultdict(set)
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
                if sab in {'CPT'}:
                    code_to_cui[code].add(cui)
                sab_for_cui[cui].add(sab)

                cui_to_terms[cui].add(str_.lower())

                if ts == "P" and ispref == "Y" and cui not in cui_to_preferred_term:
                    cui_to_preferred_term[cui] = str_

        sab_for_cui = {k: sorted(list(v)) for k, v in sab_for_cui.items()}

        # Fallback: any CUI without a captured preferred term just gets
        # whatever the first string we saw was, so downstream lookups don't
        # KeyError.
        for cui, terms in cui_to_terms.items():
            cui_to_preferred_term.setdefault(cui, list(terms)[0])

        print(f"Done. "
              # f"{len(cui_to_terms):,} unique terms, "
              f"{len(cui_to_preferred_term):,} CUIs with labels.")
        return dict(cui_to_terms), cui_to_preferred_term, code_to_cui, sab_for_cui

    @staticmethod
    def _build_graph_holder(mrrel_path, sabs=None, valid_cuis=None,
                            *,
                            use_networkkit: bool = True,
                            weighting: str = None,
                            centrality_part_cutoff: float = None,
                            cui_to_sty: Dict = None,
                            prune_skip_sty: Iterable = None,
                            ) -> GraphHolder:
        """
        Parse MRREL.RRF into an undirected networkx graph of CUIs.

        sabs: restrict to relationships asserted by these source vocabularies.
        valid_cuis: if given, only add edges where both CUIs are in this set
                    (keeps the graph aligned with a filtered term lookup).
        """
        sabs = set(sabs) if sabs else None
        # G = nx.Graph()
        # G = nx.MultiDiGraph()
        if valid_cuis is None:
            raise NotImplementedError
        G = GraphHolder.create(use_networkkit=use_networkkit,
                               valid_node_names=list(valid_cuis),
                               weighting=weighting)

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
                if valid_cuis and not all([c in valid_cuis for c in (cui1, cui2)]):
                    continue

                # Keep the most specific relationship label available on the edge.
                if not G.use_networkkit and G.has_edge(cui1, cui2):
                    prev = G.get_edge_data_from_names(cui1, cui2)
                    if {'rel': rel, 'rela': rela, 'sab': sab} in prev.values():
                        continue
                G.add_edge(cui1, cui2, rel=rel, rela=rela, sab=sab)

        G.finalize_graph(centrality_part_cutoff, cui_to_sty, prune_skip_sty)

        print(f"Done. Graph has {G.number_of_nodes():,} nodes and "
              f"{G.number_of_edges():,} edges.")
        return G

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

    def shortest_paths_between_cui_sets(self, set1: Set[str], set2: Set[str],
                                        *,
                                        cutoff: int = 5):
        return self.graph_holder.shortest_paths_between_name_sets(set1, set2, cutoff=cutoff)

    def get_cpt_lowest_snomed_predecessor(self,
                                          cui: str,
                                          ):
        if not cui.startswith('C'):
            # cui = self.code_to_cui(cui)
            raise ValueError
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
            data = self.graph.get_edge_data_from_names(cui1, cui2) or {}
            if 0 in data:
                # Just take the first description of the link.
                data = data[0]
            edges.append({
                "from_cui": cui1,
                "to_cui": cui2,
                "rel": data.get("rel"),
                "rela": data.get("rela"),
                "sab": data.get("sab"),
            })
        return edges

    def get_edges_with_data(self,
                            cui) -> List[EdgeWithData]:
        return [EdgeWithData.create(cui, p, self.graph_holder.get_edge_data_from_names(cui, p))
                for p in self.graph_holder.give_neighbors(cui)]

    def give_rel_and_rela_matches(self,
                                  cui: str,
                                  rel: None | str,
                                  rela: None | str,
                                  *,
                                  edges: List[EdgeWithData] = None) -> Iterable[EdgeWithData]:
        if edges is None:
            edges = self.get_edges_with_data(cui)

        all_v = [None if r is None else {r} for r in (rel, rela)]
        for e in edges:
            if e.has_rel_and_rela(*all_v):
                yield e


    def mine_rel(self,
                 cui: str,
                 primary_data_rel: DataRels,
                 *,
                 secondary_data_rel: DataRels = None,
                 edges: List[EdgeWithData] = None,
                 relative_depth: int = None,
                 ) -> Dict:
        use_rel = secondary_data_rel if secondary_data_rel else primary_data_rel
        out = {}
        for rel_edge in self.give_rel_and_rela_matches(cui, edges=edges, **primary_data_rel.as_dict()):
            def give_relatives(rel_edges):
                level = -1
                seen = {}
                while len(rel_edges) > 0:
                    if relative_depth is not None and level >= relative_depth:
                        break
                    level += 1

                    new_edges = []
                    for e in rel_edges:
                        if e.second_cui not in seen:
                            seen[e.second_cui] = level
                            new_edges.extend(
                                [ne for ne in self.give_rel_and_rela_matches(e.second_cui, **use_rel.as_dict())
                                 if ne.second_cui not in seen])
                    rel_edges = set(new_edges)

                return {k: v for k, v in seen.items() if v > 0}

            desc = give_relatives([rel_edge])
            if len(desc) > 0:
                out[rel_edge.second_cui] = desc

        return out

    def give_path_with_data(self,
                            source: int | str,
                            target: int | str,
                            path: Tuple[int] | Tuple[str],
                            *,
                            reverse: bool = False) -> Tuple[EdgeWithData, ...]:
        full_path = [source] + list(path) + [target]
        if reverse:
            full_path = full_path[::-1]
        if isinstance(source, int):
            full_name_path = [self._member_list[i] for i in full_path]
        elif isinstance(source, str):
            full_name_path = full_path[:]
            full_path = [self._member_lookup[n] for n in full_name_path]
        return tuple([EdgeWithData.create(f_name, s_name, self.graph_holder.get_edge_data(f, s))
                      for f, s, f_name, s_name in zip(full_path, full_path[1:], full_name_path, full_name_path[1:])])

    def get_features(self,
                     cui: str,
                     *,
                     descendant_depth: int = 3) -> Tuple[str, FrozenSet[Tuple], Dict]:
        # neighbors = self.graph_holder.give_neighbors(cui)
        # n_data = [self.graph_holder.get_edge_data_from_names(cui, n) for n in neighbors]
        cui_ind = self.graph_holder.get_ind_for_name(cui)
        # n_ind_data = [self.graph_holder.get_edge_data(cui_ind, self.graph_holder.get_ind_for_name(n)) for n in neighbors]
        immediate_paths: Set = set([])
        distant_descendants: Dict[str, Dict] = defaultdict(dict)

        bad_paths = set([])
        def is_bad(p):
            for e in range(1 + len(p)):
                if p[:e] in bad_paths:
                    return True
            return False
        # Could make parallel...
        for rel_cui_ind, raw_ind_path, path_distance in self.graph_holder.give_relatives(cui_ind, descendant_depth):
            concat = raw_ind_path + tuple([rel_cui_ind])
            if is_bad(concat):
                continue
            path: Tuple[EdgeWithData, ...] = self.give_path_with_data(cui_ind, rel_cui_ind, raw_ind_path)
            if len(path) <= 1:
                r = self.rep_for_immediate(path[0])
                if r:
                    immediate_paths.add(r)
                else:
                    bad_paths.add(concat)
            else:
                if self.acceptable_feature_path(path):
                    distant_descendants[path[0].second_cui][path[-1].second_cui] = len(path)
                else:
                    bad_paths.add(concat)

        return cui, frozenset(immediate_paths), dict(distant_descendants)

    def show_path(self,
                  edges: List[EdgeWithData]) -> str:
        out = "\n".join([f"{edges[0].first_cui} '{self.cui_to_preferred_term[edges[0].first_cui]}'"] +
                        [f"\t{e.second_cui} '{self.cui_to_preferred_term[e.second_cui]}'\t{e.all_data}"
                         for e in edges])
        return out

    @staticmethod
    def acceptable_feature_path(path: Iterable[EdgeWithData]) -> bool:
        if len(path) <= 1:
            # This feature should be immediate!
            return False
        for rel, all_data in path[0].by_rel().items():
            use_spec = UMLSCache.acceptable_specs.get(rel)
            if use_spec:
                if path[0].matches_data_rel(use_spec[0]) and all([p.matches_data_rel(use_spec[1]) for p in path[1:]]):
                    return True

        return False

    @staticmethod
    def secondary_okay_path(path: Iterable[EdgeWithData]) -> bool:
        all_rel = set([d.rel
                   for e in path
                   for d in  e.all_data])
        # No reverse direction
        if any([v in all_rel for v in {'PAR', 'RB'}]) and any([v in all_rel for v in {'CHD', 'RN'}]):
            return False

        all_rela = set([d.rela
                        for e in path
                        for d in e.all_data])

        if any([v in all_rela for v in {'do_not_code_with'}]):
            return False

        pairs = ([{f"has_{c}", f"{c}_of"}
                  for c in {'intent', 'method', 'specialty'}] +
                 [{'pathology_of_excluded', 'do_not_code_with'},
                  {'interprets', 'is_interpreted_by'},
                  {'pathology_of', 'has_direct_morphology'},
                  {'pathology_of', 'has_associated_morphology'}])
        if any([all([v in all_rela for v in p]) for p in pairs]):
            return False

        if ('pathology_of' in all_rela
                and any([v in all_rela for v in {'has_direct_morphology',  'has_associated_morphology'}])):
            return False

        for f in all_rela:
            if f.startswith('has_'):
                if f[4:] + '_of' in all_rela:
                    return False
        if any([v for v in {'CHD', 'RN'} if v in all_rel]) and any([v for v in all_rela if v.endswith('_of')]):
            pass

        if any([v for v in {'PAR', 'RB'} if v in all_rel]) and any([v for v in all_rela if v.startswith('has_')]):
            pass


        return True

    @staticmethod
    def edge_data_matches_immediate(d: DataForEdge) -> bool:
        return d.rel in {'PAR', 'CHD', 'RB', 'RN'} or \
            (d.rel in {'RO'} and d.rela not in {
                'clinician_form_of', 'has_clinician_form', 'has_consumer_friendly_form',
                'do_not_code_with', 'has_add_on_code', 'specialty_of', 'has_specialty', 'add_on_code_for',
                'consumer_friendly_form_of'
            })

    def rep_for_immediate(self, l: EdgeWithData) -> None | Tuple[str, str, str]:
        for d in l.all_data:
            if self.edge_data_matches_immediate(d):
                return d.rel, d.rela, l.second_cui
        return None

    def OLD_get_features(self,
                     cui: str,
                     *,
                     descendant_depth: int = 3) -> Tuple[str, FrozenSet[Tuple], Dict]:
        edges = self.get_edges_with_data(cui)

        def get_set(loc_edges, loc_cui):
            if len(loc_edges) == 1 and (loc_edges[0].has_rela({'has_clinician_form', 'has_consumer_friendly_form'})
                                        or loc_edges[0].has_rel({'PAR'})):
                pass
                # loc_cui = loc_edges[0].second_cui
                # loc_edges = self.get_edges_with_data(loc_cui)

            loc_out = frozenset([r for r in
                                 [self.rep_for_immediate(l) for l in loc_edges]
                                 if r])
            return loc_out, loc_edges, loc_cui

        out, edges, cui = get_set(edges, cui)

        all_rel = {}
        for pri, sec in ((DataRels('PAR', 'inverse_isa'), None),
                         (DataRels('CHD', 'isa'), None),
                         (DataRels('RB', None), DataRels('PAR', 'inverse_isa')),
                         (DataRels('RN', None), DataRels('CHD', 'isa')),
                         (DataRels('RO', None), DataRels('PAR', 'inverse_isa')), # ?? Should include?
                         # (DataRels('RO', 'procedure_site_of'), DataRels('CHD', 'isa'))
                         ):
            all_rel.update(
                self.mine_rel(cui, edges=edges[:], relative_depth=descendant_depth, primary_data_rel=pri,
                              secondary_data_rel=sec)
            )

        return cui, out, all_rel


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

class UMLS_Tracker(InventoryTracker):
    def add_entry(self,
                  entry_ind: int,
                  count_dict: Dict[str, float]):
        lex_inds = []
        cnts = []
        for k, v in count_dict.items():
            voc_ind = self._vocab.get_voc_ind(k)
            lex_inds.append(voc_ind)
            cnts.append(v)
        self.add_unweighted_counts(encoded=self.encode_item(cnts, lex_inds))
        self._label_inds.append(entry_ind)
        l = len(self._label_inds)
        if l % 10 == 0:
            print(f"l: {l}")


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
            from umls_interface import SemGroups, SemCategoryEntry
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
