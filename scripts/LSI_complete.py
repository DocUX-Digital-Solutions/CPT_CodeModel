from ml_util.inventory_tracker import spacy_holder, logger, CorpusTracker
from ml_util.cpt_holder import RawCPT

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ml_util.docux_logger import configure_logger

from ml_util.list_file import load_list_file


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop_list', type=str, default="stop_word_list.txt")
    parser.add_argument('--cpt_code_file', type=str, default='../Consolidated_Code_List.txt')
    args = parser.parse_args()
    stop_list = load_list_file(args.stop_list, min_len=1)
    # ortho_for_lemma = defaultdict(Counter)
    cpt_code_file = args.cpt_code_file

    #raw_cpt_table = RawCPT(cpt_code_file, required_init_strings=['2'])
    #raw_cpt_table = RawCPT(cpt_code_file, required_init_strings=['2'], digit_only=True)
    raw_cpt_table = RawCPT(cpt_code_file, digit_only=True)
    cpt_inventory = raw_cpt_table.give_inventory(min_form_count_per_class=1, max_similarity=5)

    # s = "Anesthesia for open or surgical arthroscopic/endoscopic procedures on distal radius, distal ulna, wrist, or hand joints; not otherwise specified"

    # f = 'Long'
    # # for f in ('Long', 'Consumer')
    # ready = [(cpt, raw_cpt_table.value_for_cpt_field(cpt, f))
    #          for cpt in raw_cpt_table.cpt_codes if self.five_d.match(cpt)
    #          ]

    ready = []
    label_inds = []
    for s, label_ind, string_ind in zip(*cpt_inventory.get_flat()):
        if string_ind == 0:
            ready.append(s)
            label_inds.append(label_ind)

    logger.info(f"label_inds: {len(label_inds)}")

    configure_logger(logger, log_file='lsi_complete.log')
    # tracker = CorpusTracker(spacy_holder, stop_list=stop_list, match_method='just_dot')
    tracker = CorpusTracker(spacy_holder, stop_list=stop_list,
                            idf_method='double_normalization',
                            class_inventory=cpt_inventory,
                            )
    tracker.add_docs(ready, label_inds)
    # print(f"unweighted_doc_matrix: {tracker.unweighted_doc_matrix.shape}")

    toy = ["rotator cuff repair", "femur fracture", "rotator cuff arthroscopy", "rotator cuff"]

    tracker.search_docs(toy, svd_dims=2000)


if __name__ == "__main__":
    main()