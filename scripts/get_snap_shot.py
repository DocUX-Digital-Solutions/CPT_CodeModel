import argparse
import os

from ml_util.modelling.random_projection import RandomProjection
from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder
from scripts.supConLearn import get_train_dev_test_dict
from src.cpt_holder import RawCPT
from src.snap_shot import SnapShot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpt_code_file', type=str, default='Consolidated_Code_List.txt')
    parser.add_argument('--required_fields', type=str, nargs='+', default=['Long', 'Consumer'])
    parser.add_argument('--init_cpt_filters', type=str, nargs='+',
                        help="Only use CPT codes which begin with one of these strings.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_file_stem", type=str, default="snapshot")
    parser.add_argument("--similarity_fn_name", type=str, default="cosine")
    parser.add_argument("--similarity_measure", type=str, default="cosine_similarity")
    parser.add_argument('--loss', type=str, default="VBATriplet")
    parser.add_argument('--per_device_train_batch_size', type=int, default=128)
    parser.add_argument('--fp16', action='store_true', help="Must be on GPU!")
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--projection_ratio', type=float, default=0.5)
    parser.add_argument('--input_dim', type=int, default=768)
    args = parser.parse_args()
    args.part_train = 1.0
    args.part_test = 0.0
    args.hard_batching = True
    args.seed = 42
    args.per_device_eval_batch_size = 128
    args.shuffle_data = False

    param_fields = {'s': args.seed,
                    'l_': args.loss,
                    'b': args.per_device_train_batch_size,
                    }
    if args.init_cpt_filters is not None and len(args.init_cpt_filters) > 0:
        param_fields['f'] = '.'.join(sorted(args.init_cpt_filters))

    output_file = (
        os.path.join(args.model_path,
                     '.'.join(
                         [args.output_file_stem] +
                         [f"{n}{v}" for n, v in param_fields.items()]
                         + ['txt'])
                     ))
    if os.path.exists(output_file):
        raise ValueError(f"Can't overwrite: {output_file}")

    raw_cpt_table = RawCPT(args.cpt_code_file,
                           required_fields=args.required_fields,
                           required_init_strings=args.init_cpt_filters)
    cpt_inventory = raw_cpt_table.give_inventory(min_form_count_per_class=len(args.required_fields))

    holder = SentenceTransformerHolder.create(args.model_path, similarity_fn_name=args.similarity_fn_name)

    dataset_dict, _, _ = get_train_dev_test_dict(cpt_inventory, args)

    l2_normalize = bool(args.similarity_measure in {"cosine_similarity"})
    if args.projection_ratio < 1.0:
        random_projection = RandomProjection(args.input_dim, int(args.projection_ratio * args.input_dim), args.seed)
    else:
        random_projection = None
    snapshot =  SnapShot(cpt_inventory, holder, dataset_dict['train'],
                         batch_size=args.per_device_eval_batch_size,
                         transform=random_projection,
                         l2_normalize=l2_normalize,
                         )

    with open(output_file, "w", encoding='utf-8') as out_H:
        out_H.write(f"args: {args}\n")
        out_H.write(f"{snapshot}")


if __name__ == "__main__":
    main()