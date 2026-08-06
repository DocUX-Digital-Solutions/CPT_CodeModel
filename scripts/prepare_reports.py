import cProfile
import pstats
import re
import torch
import numpy as np
from collections import defaultdict, Counter

from typing import Dict, FrozenSet, List

from SectionSplitter import SectionSplitter
from ml_util.reports.report_preparer import ReportPreparer
from ml_util.scispcacy_interface import SciSpacyInterface
from ml_util.umls.graph_umls import UMLSCache
from ml_util.umls.quick_umls import QuickUMLS_Matcher, UMLS_Matcher
#from datasets.packaged_modules.json.json import ujson_dumps

from ml_util.BM25_interface import BM25Index
from ml_util.modelling.faiss_interface import give_SearchIndexWrapper
from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder, SentenceCrossEncoder
from ml_util.medspacy_interface import MedSpacyHolder
from scripts.embed_descriptions import CPT_embeddings
from src.report_holder import ReportHolder


class ScoreHandler:
    def __init__(self):
        self._entries = defaultdict(dict)

    def can_add(self, n0, n1, path, dist) -> bool:
        raise NotImplementedError

    def add(self, n0, n1, path, dist):
        raise NotImplementedError

    def give_final_score(self, n0) -> float:
        raise NotImplementedError

    def get(self, n0, n1):
        try:
            return self._entries[n0][n1]
        except KeyError:
            raise


class PathScoreHandler(ScoreHandler):
        def __init__(self):
            super().__init__()

        def can_add(self, n0, n1, path, dist) -> bool:
            for_code = self._entries.get(n0)
            if for_code:
                prev_path = for_code.get(n1)
                if prev_path and len(prev_path) <= len(path):
                    return False

            return True

        def add(self, n0, n1, path, dist):
            self._entries[n0][n1] = path

        def give_final_score(self, n0) -> float:
            return float(sum([1 / (1 + len(v)) for v in self._entries[n0].values()]))


class DistanceScoreHandler(ScoreHandler):
    def __init__(self,
                 *,
                 decay_rate: float = 2.0):
        super().__init__()
        self._decay_rate = decay_rate

    def can_add(self, n0, n1, path, dist) -> bool:
        for_code = self._entries.get(n0)
        if for_code:
            prev = for_code.get(n1)
            if prev is not None and (prev[0] < dist or prev[1] < len(path)):
                return False

        return True

    def add(self, n0, n1, path, dist):
        self._entries[n0][n1] = (dist, len(path))

    def give_final_score(self, n0) -> float:
        # multi = lambda score, cnt: -np.log(score / float(cnt)) * ((1.0 / cnt) ** (1 / self._decay_rate))
        # multi = lambda score, cnt: -np.log(score / float(cnt))
        # single = lambda score: -np.log(score)
        return float(sum([1.0 / (1 + score) ** 1 / self._decay_rate
                          for score, cnt in self._entries[n0].values()]))

def main():
    import argparse
    import json


    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', type=str, required=True)
    parser.add_argument('--output_jsonl', type=str, required=True)
    parser.add_argument("--embedding_dump_dir", type=str, default="scripts/cpt_work_justBase")
    parser.add_argument('--l2_norm', action='store_true')
    parser.add_argument('--skip_sent_split', action='store_true')
    parser.add_argument('--n_nearest', type=int, default=20)
    parser.add_argument('--model_name', type=str,
                        default="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb")
    parser.add_argument('--bm25_dir', type=str)
    parser.add_argument('--bm25_weight', type=float, default=0.25)
    parser.add_argument('--cross_encoder_model', type=str)
    parser.add_argument('--cross_trust_remote', action='store_true')
    parser.add_argument('--min_retain_portion', type=float, default=0.3333)
    parser.add_argument('--umls_cache_dir', type=str,
                        default='/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/cache_2026AA')
    parser.add_argument('--quick_umls_threshold', type=float, default=0.7)
    parser.add_argument('--edge_count_cutoff', type=int, default=8)
    parser.add_argument('--scispacy_model', type=str)
    parser.add_argument('--accept_all', action='store_true')
    parser.add_argument('--use_distances', action='store_true')
    args = parser.parse_args()
    input_jsonl = args.input_jsonl
    min_retain_portion = args.min_retain_portion

    bm25_index = None
    bm25_weight = args.bm25_weight
    if args.bm25_dir is not None:
        import os
        bm25_index = BM25Index.load(os.path.join(args.bm25_dir, 'bm25_index.json'))

    report_preparer = ReportPreparer(skip_sent_split=args.skip_sent_split)
    by_odi: Dict[str, ReportHolder] = {}
    with open(input_jsonl, "r", encoding='utf-8') as in_H:
        for line in in_H:
            try:
                rh = ReportHolder.from_dict(json.loads(line.strip()))
            except:
                raise
            assert rh.operative_document_id not in by_odi
            by_odi[rh.operative_document_id] = rh
            # DIGDI
            # if len(by_odi) >= 10:
            #     break
    # medspacy_holder = MedSpacyHolder()
    section_splitter = SectionSplitter()
    umls_matcher: UMLS_Matcher  = None
    if args.scispacy_model:
        umls_matcher = SciSpacyInterface(model_name=args.scispacy_model)
    else:
        umls_matcher = QuickUMLS_Matcher(threshold=args.quick_umls_threshold)

    if args.umls_cache_dir:
        umls_cache = UMLSCache.load(args.umls_cache_dir)

        class FakeScores:
            def __init__(self):
                self._prev: Dict[FrozenSet[str], float] = dict()

            def get(self, n0, n1) -> float:
                l = frozenset([n0, n1])
                out = self._prev.get(l)
                if out is None:
                    inds = umls_cache.graph_holder.get_name_ind_tuple(n0, n1)
                    out = umls_cache.graph_holder.fake_adamic_adar_between(*inds)
                    self._prev[l] = out
                return out

            def present_dist(self, n0, n1, dist: float):
                fake = self.get(n0, n1)
                return f"d: {dist:.4f} ({fake:.4f} -> {dist - fake:+.4f})"

        fake_scores = FakeScores()
    else:
        umls_cache = None
        fake_scores = None

    l2_norm = args.l2_norm
    n_nearest = args.n_nearest
    cpt_embeddings = CPT_embeddings.load(args.embedding_dump_dir, l2_norm=l2_norm)
    search_index = give_SearchIndexWrapper(cpt_embeddings.embeddings,
                                           similarity_measure='cosine_similarity' if l2_norm else 'ref_norm')
    model_holder = SentenceTransformerHolder.create(model_name=args.model_name)
    recall_by_cpt = defaultdict(Counter)
    cross_encoder = None if args.cross_encoder_model is None \
        else SentenceCrossEncoder.create(args.cross_encoder_model,
                                         trust_remote_code=args.cross_trust_remote)

    def get_score_handler() -> ScoreHandler:
        return DistanceScoreHandler() if args.use_distances else PathScoreHandler()

    with open(args.output_jsonl, "w", encoding='utf-8') as out_H:
        PROFILE = False
        if True:
        #PROFILE = True
        # with cProfile.Profile() as pr:
            all_odi = sorted(list(by_odi.keys()))
            for odi in all_odi:
                print(f"odi: {odi}")
                raw_doc = by_odi[odi].pdf_text
                texts = []
                # do_purge = True
                do_purge = False

                def give_simple():
                    spans_with_pre = [s for s in report_preparer.give_spans_with_pre(raw_doc)]
                    return [raw_doc[s.text_start:s.text_end]
                            for s in spans_with_pre]

                if not do_purge:
                    texts = give_simple()
                else:
                    offs, stretches = section_splitter.purge_skip_sections(raw_doc)
                    if sum([len(s) for s in stretches]) < min_retain_portion * len(raw_doc):
                        texts = give_simple()
                        print(f"Got less than {min_retain_portion} for: {odi}.")
                    else:
                        for off, stretch in zip(*section_splitter.purge_skip_sections(raw_doc)):
                            for s in report_preparer.give_spans_with_pre(stretch):
                                texts.append(raw_doc[off + s.text_start:off + s.text_end])

                doc_embeddings = torch.cat([b for b in model_holder.encode_no_grad(texts)])
                if l2_norm:
                    doc_embeddings = torch.nn.functional.normalize(doc_embeddings)
                distances, indices = search_index.search(doc_embeddings, n_nearest=n_nearest)
                if cross_encoder is not None:
                    new_sim = []
                    new_indices = []
                    for t_ind, t, in enumerate(texts):
                        cross_scores = cross_encoder.score_queries(
                            queries=[cpt_embeddings.descriptions[i] for i in indices[t_ind]],
                            candidate=t)
                        cs_as = (-1 * cross_scores).argsort()
                        new_sim.append(cross_scores[cs_as])
                        new_indices.append(indices[t_ind][cs_as])
                    distances, indices = [np.stack(l) for l in (new_sim, new_indices)]

                distances, ids, indices = [torch.tensor(a) for a in cpt_embeddings.id_compress(distances, indices)]
                if bm25_index is not None:
                    d, i = bm25_index.search(texts)
                    bm_distances, bm_ids, bm_indices = [torch.tensor(a) for a in bm25_index.id_compress(d, i)]
                    distances, ids = \
                        cpt_embeddings.min_max_interpolate(
                            all_distances=[distances, bm_distances],
                            all_ids=[ids, bm_ids],
                            add_weights=[bm25_weight])

                if umls_matcher is not None:
                    umls_matches = {t[0]: t for t in umls_matcher.give_matches(raw_doc).by_cui()}
                    # how to use indices??
                    index_matches = [
                        cpt_cui
                        for id in ids.unique().tolist() if id >= 0
                        for cpt_cui in umls_cache.code_to_cui.get(cpt_embeddings.label_for_id(id), [])]

                    inputs = [umls_cache.graph_holder.filter_illegal_cuis(set(d))
                              for d in (list(umls_matches.keys()), index_matches)]
                    print(f"input counts: {[len(i) for i in inputs]}")
                    paths_to_cpts = umls_cache.shortest_paths_between_cui_sets(*inputs, cutoff=args.edge_count_cutoff)

                    loc_cui_to_cpts = defaultdict(list)
                    for pc in by_odi[odi].procedure_combinations:
                        cpt = pc.cpt4Code
                        if cpt not in umls_cache.code_to_cui:
                            print(f"No cui for CPT: {cpt} '{umls_cache.cui_to_preferred_term.get(cpt, None)}'")
                        else:
                            for cui in umls_cache.code_to_cui.get(cpt, []):
                                loc_cui_to_cpts[cui].append(cpt)

                    def give_ref_matches(cpt_id):
                        loc_cpt_matches = loc_cui_to_cpts.get(cpt_id)
                        if loc_cpt_matches:
                            for_cpt = f"report_cpts: {' '.join(loc_cpt_matches)}"
                        else:
                            for_cpt = 'NO CPT MATCHES'
                        return for_cpt

                    # paths_code_to_umls = defaultdict(dict)
                    score_handler = get_score_handler()
                    for umls_id, for_umls_id in paths_to_cpts.items():
                        print(
                            f"umls_id: {umls_id} "
                            f"{umls_cache.cui_to_preferred_term.get(umls_id, 'NO DESCRIPTION FOUND!')}")
                        skipped = []
                        for cpt_id, (dist, path) in for_umls_id.items():
                            code = umls_cache.cui_to_code[cpt_id]
                            okay = True
                            acceptable = True
                            path_with_data = None
                            if len(path) > 0:
                                path_with_data = umls_cache.give_path_with_data(umls_id, cpt_id, path, reverse=True)
                                okay = umls_cache.secondary_okay_path(path_with_data)
                                if len(path_with_data) == 1:
                                    if umls_cache.rep_for_immediate(path_with_data[0]) is None:
                                        acceptable = False
                                        # skipped.append((dist, path_with_data))
                                        # continue
                                else:
                                    if not umls_cache.acceptable_feature_path(path_with_data):
                                        acceptable = False
                                        # skipped.append((dist, path_with_data))
                                        # continue
                            if not okay and not args.accept_all:
                                skipped.append((dist, path_with_data, acceptable))
                                continue

                            if not score_handler.can_add(code, umls_id, path, dist):
                                continue
                            # for_code = paths_code_to_umls.get(code)
                            # if for_code:
                            #     prev_path = for_code.get(umls_id)
                            #     if prev_path and len(prev_path) <= len(path):
                            #         continue

                            print(f"USE:\tacceptable: {acceptable}\td: {fake_scores.present_dist(umls_id, cpt_id, dist)}"
                                  f"\t{give_ref_matches(cpt_id)} "
                                  f"{umls_cache.show_path(path_with_data) if path_with_data else 'DIRECT'}")
                            # paths_code_to_umls[code][umls_id] = path
                            score_handler.add(code, umls_id, path, dist)
                        # for s in sorted(skipped, key=lambda x: (len(x), x[0].first_cui, x[0].second_cui)):
                        for dist, s, acceptable in sorted(skipped, key=lambda x:
                        (x[0], len(x[1]), x[1][0].first_cui, x[1][0].second_cui)):
                            # okay = umls_cache.secondary_okay_path(s)
                            print(f"SKIPPED:\t{fake_scores.present_dist(umls_id, cpt_id, dist)}\tacceptable: {acceptable}\t"
                                  f"{give_ref_matches(s[0].first_cui)}\t"
                                  f"{umls_cache.show_path(s)}")
                            pass
                        pass
                    pass

                # def sum_inv_for_cpt(cpt_code):
                #     return sum([1 / (1 + len(v)) for v in paths_code_to_umls[cpt_code].values()])

                for pc in by_odi[odi].procedure_combinations:
                    first_desc, matches = cpt_embeddings.give_matches(pc.cpt4Code, ids)
                    print(f"cpt: {pc.cpt4Code} '{first_desc}' cuis: {umls_cache.code_to_cui.get(pc.cpt4Code, None)} "
                          f"umls_inv: {score_handler.give_final_score(pc.cpt4Code):.2f}")
                    if first_desc is None or len(matches) < 1:
                        print(f"no match for {pc.cpt4Code} in {odi}.")
                        recall_by_cpt[pc.cpt4Code][-1] += 1
                        # Show the top matches...
                    else:
                        by_rank = torch.argsort(matches[:, 1])
                        by_score = torch.argsort(-1 * torch.tensor(distances)[matches[:, 0], matches[:, 1]])
                        # Just take the mean???
                        by_combined = (by_rank + by_score).argsort()
                        # use_for_sort = by_combined
                        use_for_sort = by_rank
                        # if by_max
                        # use_for_sort = by_score
                        for m_rank, m in enumerate(matches[use_for_sort]):
                            if m_rank == 0:
                                recall_by_cpt[pc.cpt4Code][m[1].item()] += 1
                            print(f"{odi}\t{pc.cpt4Code} m: {m_rank} "
                                  f"rank for string: {m[1]} "
                                  f"dist: {distances[m[0], m[1]]:.3f} "                              
                                  f"{texts[m[0]]}")

                            for rank, (score, cpt_id) in enumerate(zip(distances[m[0]][:m[1]], ids[m[0]][:m[1]])):
                                label = cpt_embeddings.label_for_id(cpt_id)
                                print(f"\tover at {rank} ({score:.3f}): "                                  
                                      f"{label} "
                                      f"umls_inv: {score_handler.give_final_score(label):.2f} "
                                      f"{cpt_embeddings.first_description_for_id(cpt_id)}")
                    pass
                if PROFILE:
                    stats = pstats.Stats(pr)
                    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)  # Prints top 10 bottlenecks
                pass

                # Need to check...

    digits_only = re.compile(r"^[0-9]{5}$")
    view_inds = (1, 3, 5, 10, 15, 20, 30, 40)
    by_type = {k: 0 for k in view_inds}
    by_instance = {k: 0 for k in view_inds}
    total_instances = 0
    for cpt_code, tally in sorted(recall_by_cpt.items()):
        if digits_only.match(cpt_code) is None:
            continue
        total = sum([v for v in tally.values()])
        print(f"cpt: {cpt_code} total: {total}")
        total = float(total)
        total_instances += total
        fr = tally.get(-1, 0)
        print(f"FR:\t{fr} ({fr/total:.2f}%)")
        cum_vals = []
        cum = 0
        for k, v in sorted(tally.items()):
            if k >= 0:
                cum += v
                cum_vals.append((k, cum))
        c_ind = -1
        for v_ind in view_inds:
            while c_ind + 1 < len(cum_vals) and cum_vals[c_ind + 1][0] < v_ind:
                c_ind += 1
            loc = 0 if c_ind == -1 else cum_vals[c_ind][1]
            print(f"{v_ind}\t{loc} ({(100*loc) / total:.2f}%)")
            if loc > 0:
                by_type[v_ind] += 1
                by_instance[v_ind] += loc
    total_types = float(len(recall_by_cpt))

    print(f"total_types: {total_types} total_instances: {total_instances}")

    print(f"SUM by view_ind")
    for v_ind in view_inds:
        print(f"{v_ind}"
              f" types: {by_type[v_ind]} ({(100 * by_type[v_ind]) / total_types:.2f}%)"
              f" instances: {by_instance[v_ind]} ({(100 * by_instance[v_ind]) / total_instances:.2f}%)")


if __name__ == '__main__':
    main()
