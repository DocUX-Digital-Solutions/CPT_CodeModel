import argparse
import logging
from src.cpt_holder import RawCPT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpt_code_file', type=str, default='Consolidated_Code_List.txt')
    parser.add_argument('--required_fields', type=str, nargs='+', default=['Long', 'Consumer'])
    parser.add_argument('--init_cpt_filters', type=str, nargs='+',
                        help="Only use CPT codes which begin with one of these strings.")
    args = parser.parse_args()

    raw_cpt_table = RawCPT(args.cpt_code_file,
                           required_fields=args.required_fields,
                           required_init_strings=args.init_cpt_filters)