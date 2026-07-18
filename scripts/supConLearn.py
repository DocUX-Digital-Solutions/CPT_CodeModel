import argparse
import logging
import os, shutil

import torch
from datasets import DatasetDict

from typing import Dict, Type, Tuple

from ml_util.modelling.random_projection import RandomProjection
from ml_util.cpt_holder import get_raw_code_table
from ml_util.cpt_holder import supported_tasks as supported_raw_tasks
from ml_util.random_utils import set_seed
from ml_util.classes import ClassInventory
from ml_util.docux_logger import give_logger, configure_logger
from ml_util.modelling.supervised_contrastive import SentenceTransformerSupConTrainer
from ml_util.modelling.batch_all import get_BatchAll_train_dev_test_dict, BatchCache, give_ranges_by_common
from ml_util.modelling.triplet import get_Triplet_train_dev_test_dict, SentenceTransformerTripletTrainer, \
    SentenceTransformerAllBatchTripletTrainer
from ml_util.modelling.sentence_transformer_interface import SentenceTransformerCustomTrainer
from src.snap_shot import SnapShot
from ml_util.modelling.faiss_interface import BaseIndexWrapper

logger: logging.Logger = None

supported_loss = ('SupCon', 'Triplet', 'BATriplet', 'BShATriplet', 'VBATriplet', 'VBAPair')


def get_train_dev_test_dict(code_inventory: ClassInventory,
                            args: argparse.PARSER,
                            *,
                            similarity_measure: str = 'cosine_similarity',
                            ) -> Tuple[DatasetDict, BatchCache, BatchCache]:
    if args.bf16:
        vector_dtype = torch.bfloat16
    elif args.fp16:
        vector_dtype = torch.float16
    else:
        vector_dtype = torch.float32
    label_dtype = code_inventory.give_torch_dtype()

    if args.hard_batching:
        l2_normalize = bool(args.similarity_measure in ('cosine_similarity'))
        if args.projection_ratio < 1.0:
            random_projection = RandomProjection(args.input_dim, int(args.projection_ratio * args.input_dim), args.seed)
        else:
            random_projection = None
        train_batch_cache = BatchCache(args.per_device_train_batch_size, vector_dtype, label_dtype,
                                       transform=random_projection, l2_normalize=l2_normalize)
        eval_batch_cache = BatchCache(args.per_device_eval_batch_size, vector_dtype, label_dtype,
                                      transform=random_projection, l2_normalize=l2_normalize)
    else:
        train_batch_cache = None
        eval_batch_cache = None

    loc_args = ()
    loc_kwargs = {'class_inventory': code_inventory,
                  'part_train': args.part_train,
                  'part_test': args.part_test,
                  'shuffle': args.shuffle_data,
                  'seed': args.seed,
                  'hard_batching': args.hard_batching,
                  'train_batch_size': args.per_device_train_batch_size,
                  'test_batch_size': args.per_device_eval_batch_size,
                  }

    def give_dict() -> DatasetDict:
        if args.loss in ('SupCon'):
            return get_BatchAll_train_dev_test_dict(*loc_args, **loc_kwargs)
        elif args.loss == 'Triplet':
            return get_Triplet_train_dev_test_dict(*loc_args, **loc_kwargs)
        elif args.loss in ('BATriplet', 'BShATriplet', 'VBATriplet', 'VBAPair'):
            return get_BatchAll_train_dev_test_dict(*loc_args,
                                                    train_batch_cache=train_batch_cache,
                                                    **loc_kwargs,
                                                    )
        else:
            raise NotImplementedError

    return give_dict(), train_batch_cache, eval_batch_cache


trainer_class_map: Dict[str, Type] = \
    {'SupCon': SentenceTransformerSupConTrainer,
     'Triplet': SentenceTransformerTripletTrainer,
     'BATriplet': SentenceTransformerAllBatchTripletTrainer,
     'BShATriplet': SentenceTransformerAllBatchTripletTrainer,
     'VBATriplet': SentenceTransformerAllBatchTripletTrainer,
     'VBAPair': SentenceTransformerAllBatchTripletTrainer,
     }


def get_trainer(args: argparse.PARSER,
                class_inventory: ClassInventory = None) -> SentenceTransformerCustomTrainer:
    dataset_dict, train_batch_cache, eval_batch_cache = get_train_dev_test_dict(class_inventory, args)
    loc_kwargs = {'top_args': args,
                  'model_name': args.model_name,
                  'train_dataset': dataset_dict['train'],
                  'eval_dataset': dataset_dict.get('valid', None),
                  'loss_name': args.loss,
                  'class_inventory': class_inventory,
                  'train_batch_cache': train_batch_cache,
                  'eval_batch_cache': eval_batch_cache,
                  'logging_dir': os.path.join(args.output_dir, 'logs'),
                  'logger': logger,
                  }
    trainer_class = trainer_class_map[args.loss]

    trainer = trainer_class(**loc_kwargs)

    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', type=str, default='Consolidated_Code_List.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str,
                        default="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb")
    parser.add_argument('--output_dir_stem', type=str, required=True)
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--per_device_train_batch_size', type=int, default=128)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=5e-05)
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--part_train', type=float, default=0.9)
    parser.add_argument('--part_test', type=float, default=0.05)
    parser.add_argument('--shuffle_data', action='store_true')
    parser.add_argument('--init_cpt_filters', type=str, nargs='+',
                        help="Only use CPT codes which begin with one of these strings.")
    parser.add_argument('--loss_temperature', type=float, default=0.01)
    parser.add_argument('--evaluation_strategy', type=str, default='epoch')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--log_level', type=str, default='DEBUG')
    parser.add_argument('--loss', type=str, default='SupCon', choices=supported_loss)
    parser.add_argument('--required_fields', type=str, nargs='*')
                        # default=['Long', 'Consumer'])
    parser.add_argument('--triplet_loss_margin', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--output_hidden_states', action='store_true')
    parser.add_argument('--hard_batching', action='store_true',
                        help="Keep model output embeddings for subsequent hard batching.")
    parser.add_argument('--optim', type=str, default='adamw_torch')
    parser.add_argument('--fp16', action='store_true', help="Must be on GPU!")
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument("--torch_empty_cache_steps", type=int, default=1 ,
                        help="Needed, at least, when running of a MacBook with 24GB physical RAM (MPS).")
    parser.add_argument('--allow_overwrite', action='store_true')
    parser.add_argument('--reload_dataloaders_every_n_epochs', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--projection_ratio', type=float, default=0.5)
    parser.add_argument('--similarity_measure', type=str, default='cosine_similarity',
                        choices=BaseIndexWrapper.supported_similarity_measures)
    parser.add_argument('--save_strategy', default='epoch')
    parser.add_argument('--task_type', type=str, default='CPT', choices=supported_raw_tasks )
    # parser.add_argument("--device", type=str, default='mps')
    args = parser.parse_args()

    param_fields = {'s': args.seed,
                    'l_': args.loss,
                    'b': args.per_device_train_batch_size,
                    'lr': args.learning_rate,
                    'e': args.num_train_epochs,
                    'tr': args.part_train,
                    'tst': args.part_test,
                    'wr': args.warmup_ratio,
                    'wd': args.weight_decay,
                    'o_': args.optim,
                    'gn_': args.max_grad_norm}

    if args.init_cpt_filters is not None and len(args.init_cpt_filters) > 0:
        param_fields['f'] = '.'.join(sorted(args.init_cpt_filters))

    args.output_dir = '.'.join(
        [args.output_dir_stem] + [args.task_type] +
        [f"{n}{v}" for n, v in param_fields.items()]
    )
    if args.bf16:
        args.output_dir  = args.output_dir + '.bf16'
    if args.fp16:
        args.output_dir  = args.output_dir + '.fp16'

    args.log_file = os.path.join(args.output_dir, 'logs', 'main.log')
    if os.path.exists(args.output_dir):
        if args.allow_overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise ValueError(f"Already exists: {args.output_dir}; overwrite is not enabled.")
    os.makedirs(os.path.join(args.output_dir, 'logs'))
    global logger
    logger = give_logger()
    configure_logger(logger, log_file=args.log_file, level=args.log_level)
    logger.info(f"args: {args}")

    set_seed(args.seed)

    # For the trainer
    args.do_train = True
    args.do_eval = True
    args.do_predict = False
    args.eval_strategy = 'epoch'

    raw_code_table = get_raw_code_table(args.code_file,
                                       task_type=args.task_type,
                                       required_fields=args.required_fields,
                                       required_init_strings=args.init_cpt_filters)
    logger.info(f"raw code cnt: {len(raw_code_table.by_code)}")
    code_inventory = raw_code_table.give_inventory(min_form_count_per_class=len(args.required_fields),
                                                   name=f"{args.task_type} Inventory",
                                                   max_similarity=raw_code_table.target_len)

    trainer = get_trainer(args, code_inventory)
    init_metrics = trainer.evaluate()
    snapshot_kwargs = {'batch_size': args.per_device_train_batch_size,
                       'l2_normalize': bool(args.similarity_measure in ('cosine_similarity'))}
    if trainer.train_batch_cache is not None:
        snapshot_kwargs.update(
            {'transform': trainer.train_batch_cache.transform}
        )
    init_snapshot = SnapShot(code_inventory, trainer.holder, trainer.train_dataset, **snapshot_kwargs)
    logger.info(f"init_metrics: {init_metrics}")
    trainer.train()
    final_snapshot = SnapShot(code_inventory, trainer.holder, trainer.train_dataset, **snapshot_kwargs)

    final_snapshot.compare_to_prev(init_snapshot)
    logger.info(f"{give_ranges_by_common()}")


if __name__ == '__main__':
    main()
