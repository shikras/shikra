import os
import sys
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple
from argparse import SUPPRESS

import datasets
import transformers
from mmengine.config import Config, DictAction
from transformers import HfArgumentParser, set_seed, add_start_docstrings
from transformers import Seq2SeqTrainingArguments as HFSeq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@dataclass
@add_start_docstrings(HFSeq2SeqTrainingArguments.__doc__)
class Seq2SeqTrainingArguments(HFSeq2SeqTrainingArguments):
    do_multi_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the multi-test set."})


def prepare_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')

    hf_parser = HfArgumentParser((Seq2SeqTrainingArguments,))
    hf_parser, required = block_required_error(hf_parser)

    args, unknown_args = parser.parse_known_args(args)
    known_hf_args, unknown_args = hf_parser.parse_known_args(unknown_args)
    if unknown_args:
        raise ValueError(f"Some specified arguments are not used "
                         f"by the ArgumentParser or HfArgumentParser\n: {unknown_args}")

    # load 'cfg' and 'training_args' from file and cli
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    training_args = cfg.training_args
    training_args.update(vars(known_hf_args))

    # check training_args require
    req_but_not_assign = [item for item in required if item not in training_args]
    if req_but_not_assign:
        raise ValueError(f"Requires {req_but_not_assign} but not assign.")

    # update cfg.training_args
    cfg.training_args = training_args

    # initialize and return
    training_args = Seq2SeqTrainingArguments(**training_args)
    training_args = check_output_dir(training_args)

    # logging
    if is_main_process(training_args.local_rank):
        to_logging_cfg = Config()
        to_logging_cfg.model_args = cfg.model_args
        to_logging_cfg.data_args = cfg.data_args
        to_logging_cfg.training_args = cfg.training_args
        logger.info(to_logging_cfg.pretty_text)

    # setup logger
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.logging.set_verbosity(log_level)
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()
    # setup_print_for_distributed(is_main_process(training_args))

    # Log on each process the small summary:
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"  distributed training: {bool(training_args.local_rank != -1)}, fp16 training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    return cfg, training_args


def block_required_error(hf_parser: HfArgumentParser) -> Tuple[HfArgumentParser, List]:
    required = []
    # noinspection PyProtectedMember
    for action in hf_parser._actions:
        if action.required:
            required.append(action.dest)
        action.required = False
        action.default = SUPPRESS
    return hf_parser, required


def check_output_dir(training_args):
    # Detecting last checkpoint.
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return training_args


if __name__ == "__main__":
    _ = prepare_args()
