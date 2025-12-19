import argparse
import sys
from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder
from src.cpt_holder import RawCPT
from src.snap_shot import Searcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--required_fields', type=str, nargs='+', default=['Long', 'Consumer'])
    parser.add_argument('--init_cpt_filters', type=str, nargs='+',
                        help="Only use CPT codes which begin with one of these strings.")
    parser.add_argument('--cpt_code_file', type=str, default='Consolidated_Code_List.txt')
    args = parser.parse_args()

    raw_cpt_table = RawCPT(args.cpt_code_file,
                           required_fields=args.required_fields,
                           required_init_strings=args.init_cpt_filters)
    cpt_inventory = raw_cpt_table.give_inventory(min_form_count_per_class=len(args.required_fields))

    model_holder = SentenceTransformerHolder.create(args.model_dir)

    searcher = Searcher(cpt_inventory, model_holder)

    print(f"ready for input!")

    for line in sys.stdin:
        line = line.strip()
        if len(line) < 1:
            continue
        if line.lower() in ('stop', 'quit', 'exit'):
            break
        query = line
        print(f"query: {query}")
        raw_out = searcher.search(query)
        raw_out = "\n".join(raw_out)

        print(f'input: {query.strip()}\n{raw_out}')


if __name__ == '__main__':
    main()