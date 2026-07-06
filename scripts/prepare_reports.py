import re
from dataclasses import dataclass
import torch
from collections import defaultdict, Counter

from typing import List, Tuple, Dict

from ml_util.modelling.faiss_interface import SearchIndexWrapper, give_SearchIndexWrapper
from ml_util.sentence_breaks import SentenceBreaker
from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder
from scripts.embed_descriptions import CPT_embeddings
from src.report_holder import ReportHolder


@dataclass(frozen=True)
class ChunkSpans:
    prefix_start: int
    prefix_end: int
    text_start: int
    text_end: int

    @property
    def text_length(self) -> int:
        return self.text_end - self.text_start

    @classmethod
    def merge(cls,
              chunks: List['ChunkSpans']):
        if len(chunks) <= 1:
            raise ValueError
        chunks = sorted(chunks, key=lambda x: x.text_start)
        return cls(chunks[0].prefix_start, chunks[0].prefix_end,
                   chunks[0].text_start, chunks[-1].text_end)

    def texts(self,
              raw_text: str) -> Tuple[str, str]:
        return raw_text[self.prefix_start:self.prefix_end], raw_text[self.text_start:self.text_end]


class ReportPreparer:
    hr_re = re.compile(r"\n+")

    def __init__(self,
                 *,
                 skip_sent_split: bool = False,
                 simple_merge_left: int = 20,
                 simple_merge_right: int = 200):
        self.skip_sent_split = skip_sent_split
        if not skip_sent_split:
            self.sentence_breaker = SentenceBreaker.build()

        self.simple_merge_left = simple_merge_left
        self.simple_merge_right = simple_merge_right

    def give_hr_bound_spans(self,
                            full_text: str) -> List[Tuple[int]]:
        hr_spans = [m.regs[0] for m in self.hr_re.finditer(full_text)]
        out = []
        if len(hr_spans) < 1:
            out.append((0, len(full_text)))
        else:
            out = [(0, hr_spans[0][0])]
            for ind in range(len(hr_spans) -1):
                out.append((hr_spans[ind][1], hr_spans[1+ind][0]))
            out.append(
                (hr_spans[-1][1], len(full_text))
            )
        return out

    def give_plain_spans_with_pre(self,
                                  raw_doc) -> List[ChunkSpans]:
        hr_bound_spans = self.give_hr_bound_spans(raw_doc)
        ready_spans = [raw_doc[s[0]:s[1]] for s in hr_bound_spans]
        all_sent_spans = [None for _ in ready_spans] if self.skip_sent_split else self.sentence_breaker.give_sentence_span_list(ready_spans)
        out: List[ChunkSpans] = []
        for hr_bound_ind, (hr_bound_span, sent_spans) in enumerate(zip(hr_bound_spans, all_sent_spans)):
            pre_begin = 0 if hr_bound_ind == 0 else hr_bound_spans[-1 + hr_bound_ind][1]
            if self.skip_sent_split:
                out.append(
                    ChunkSpans(pre_begin, *[hr_bound_span[i] for i in (0, 0, 1)])
                )
            else:
                for sent_span in sent_spans:
                    out.append(
                        ChunkSpans(pre_begin,
                                   *[hr_bound_span[0] + sent_span[i] for i in (0, 0, 1)]
                                   )
                    )
                    pre_begin = hr_bound_span[0] + sent_span[1]

        return out

    def merge_short_hr(self,
                       raw_doc: str,
                       spans_with_pre: List[ChunkSpans]):
        # Just greedy, left to right
        start_ind = -2 + len(spans_with_pre)
        for ind in range(start_ind, 0, -1):
            if spans_with_pre[ind].text_length <= self.simple_merge_left \
                    and spans_with_pre[1 + ind].text_length <= self.simple_merge_right \
                    and spans_with_pre[ind + 1].texts(raw_doc)[0] == "\n":
                spans_with_pre[ind] = ChunkSpans.merge(spans_with_pre[ind:ind + 2])
                spans_with_pre.pop(1 + ind)
            else:
                pass

    def give_spans_with_pre(self,
                            raw_text: str) -> List[ChunkSpans]:
        out = self.give_plain_spans_with_pre(raw_text)
        # self.merge_short_hr(raw_text, out)

        return out


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
    args = parser.parse_args()
    input_jsonl = args.input_jsonl

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

    l2_norm = args.l2_norm
    n_nearest = args.n_nearest
    cpt_embeddings = CPT_embeddings.load(args.embedding_dump_dir, l2_norm=l2_norm)
    search_index = give_SearchIndexWrapper(cpt_embeddings.embeddings,
                                           similarity_measure='cosine_similarity' if l2_norm else 'ref_norm')
    model_holder = SentenceTransformerHolder.create(model_name=args.model_name)
    recall_by_cpt = defaultdict(Counter)
    with open(args.output_jsonl, "w", encoding='utf-8') as out_H:
        all_odi = sorted(list(by_odi.keys()))
        for odi in all_odi:
            print(f"odi: {odi}")
            raw_doc = by_odi[odi].pdf_text
            spans_with_pre = [s for s in report_preparer.give_spans_with_pre(raw_doc)]
            texts = [raw_doc[s.text_start:s.text_end]
                     for s in spans_with_pre]
            doc_embeddings = torch.cat([b for b in model_holder.encode_no_grad(texts)])
            if l2_norm:
                doc_embeddings = torch.nn.functional.normalize(doc_embeddings)
            distances, indices = search_index.search(doc_embeddings, n_nearest=n_nearest)
            distances, ids, indices = [torch.tensor(a) for a in cpt_embeddings.id_compress(distances, indices)]
            for pc in by_odi[odi].procedure_combinations:
                try:
                    target_id = cpt_embeddings.id_for_label(pc.cpt4Code)
                except ValueError:
                    continue
                print(f"cpt {pc.cpt4Code}: "
                      f"{cpt_embeddings.first_description_for_id(target_id)}")
                matches = torch.nonzero(ids == target_id)
                if matches.shape[0] < 1:
                    print(f"no match for {pc.cpt4Code} in {odi}.")
                    recall_by_cpt[pc.cpt4Code][-1] += 1
                else:
                    by_rank = torch.argsort(matches[:, 1])
                    by_score = torch.argsort(-1 * torch.tensor(distances)[matches[:, 0], matches[:, 1]])
                    # Just take the mean???
                    by_combined = (by_rank + by_score).argsort()
                    # use_for_sort = by_combined
                    use_for_sort = by_rank
                    for m_rank, m in enumerate(matches[use_for_sort]):
                        if m_rank == 0:
                            recall_by_cpt[pc.cpt4Code][m[1].item()] += 1
                        print(f"{odi}\t{pc.cpt4Code} m: {m_rank} "
                              f"rank for string: {m[1]} "
                              f"dist: {distances[m[0], m[1]]:.3f} "
                              f"{texts[m[0]]}")

                        for rank, (score, cpt_id) in enumerate(zip(distances[m[0]][:m[1]], ids[m[0]][:m[1]])):
                            print(f"\tover at {rank} ({score:.3f}): "
                                  f"{cpt_embeddings.label_for_id(cpt_id)} "
                                  f"{cpt_embeddings.first_description_for_id(cpt_id)}")
                pass

            # Need to check...

    view_inds = (1, 3, 5, 10, 15, 20, 30, 40)
    by_type = {k: 0 for k in view_inds}
    by_instance = {k: 0 for k in view_inds}
    total_instances = 0
    for cpt_code, tally in sorted(recall_by_cpt.items()):
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

    print(f"SUM by view_ind")
    for v_ind in view_inds:
        print(f"{v_ind}"
              f" types: {by_type[v_ind]} ({(100 * by_type[v_ind]) / total_types:.2f}%)"
              f" instances: {by_instance[v_ind]} ({(100 * by_instance[v_ind]) / total_instances:.2f}%)")


if __name__ == '__main__':
    main()
