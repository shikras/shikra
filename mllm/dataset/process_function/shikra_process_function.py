import sys
import copy
import warnings
import logging
from typing import Dict, Any, List

import PIL.Image
import torch
from PIL import Image
from transformers import LlamaTokenizer

from ..root import (
    FUNCTIONS,
    IMAGE_PLACEHOLDER,
    BaseImageProcessFunc,
    BaseConvProcessFunc,
    BaseTextProcessFunc,
)
from ...conversation import SeparatorStyle, Conversation

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = IMAGE_PLACEHOLDER
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@FUNCTIONS.register_module()
class ShikraConvProcess(BaseConvProcessFunc):
    def __call__(self, raw_conv: List[Dict[str, Any]], preprocessor: Dict[str, Any], conv_template: Conversation) -> List[Dict[str, Any]]:
        conv_processor_cfg = preprocessor['conv']

        image_token_len = conv_processor_cfg['image_token_len']
        sep_image_conv_front = conv_processor_cfg.get('sep_image_conv_front', False)
        use_im_start_end = conv_processor_cfg.get('use_im_start_end', False)
        # assert DEFAULT_IMAGE_PATCH_TOKEN in preprocessor['text'].get_vocab()
        # if use_im_start_end:
        #     assert DEFAULT_IM_START_TOKEN in preprocessor['text'].get_vocab()
        #     assert DEFAULT_IM_END_TOKEN in preprocessor['text'].get_vocab()

        if sep_image_conv_front:
            raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            raw_conv[0]['value'] = DEFAULT_IMAGE_TOKEN + conv_template.sep + conv_template.roles[0] + ": " + raw_conv[0]['value']
        for sentence in raw_conv:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

        return raw_conv


@FUNCTIONS.register_module()
class ShikraTextProcess(BaseTextProcessFunc):

    def __call__(self, conv: Conversation, preprocessor: Dict[str, Any], mode: str, **tokenize_kwargs) -> Dict[str, Any]:
        tokenizer = preprocessor['text']
        assert isinstance(tokenizer, LlamaTokenizer), "only work for LlamaTokenizer"

        _truncation_size = tokenize_kwargs.pop('truncation_size', None)
        _kwargs = {'return_tensors': 'pt'}
        _kwargs.update(tokenize_kwargs)

        if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
            if mode in ['train']:
                ret = self.tk_conv_colon_two_train(conv, tokenizer, **_kwargs)
            else:
                ret = self.tk_conv_colon_two_eval(conv, tokenizer, **_kwargs)
        else:
            raise ValueError(f"unrecognized conv_style: {conv.sep_style}.\n the conv is {conv}")

        if _truncation_size is None:
            return ret
        if len(ret['input_ids']) <= _truncation_size:
            return ret

        origin_len = len(ret['input_ids'])
        ids_to_remove_num = origin_len - _truncation_size
        # truncation. should carefully not truncate <img_token>
        ids_should_not_remove = list(map(
            tokenizer.convert_tokens_to_ids,
            (DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN)
        ))
        back_no_image = all(ids not in ids_should_not_remove for ids in ret['input_ids'][_truncation_size:])
        if back_no_image:
            tgt_ids = list(range(_truncation_size))
        else:
            ids_to_remove = set()
            for idx in range(origin_len - 1, -1, -1):
                if ret['input_ids'][idx] not in ids_should_not_remove:
                    ids_to_remove.add(idx)
                    if len(ids_to_remove) >= ids_to_remove_num:
                        break
            tgt_ids = [_ for _ in range(origin_len) if _ not in ids_to_remove]
        logger.warning(f"truncate sample size from {origin_len} to {len(tgt_ids)}.")
        assert len(tgt_ids) == _truncation_size, f"{len(tgt_ids)}, {_truncation_size}, {ret['input_ids'].tolist()}"
        truncated_ret = {k: v[tgt_ids] for k, v in ret.items()}
        return truncated_ret

    # noinspection PyMethodMayBeStatic
    def tk_conv_colon_two_train(self, conv, tokenizer, **kwargs):
        conversation = conv.get_prompt()
        input_ids = tokenizer([conversation, ], **kwargs).input_ids[0]
        target = copy.deepcopy(input_ids)
        assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # <s> <space>
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                warnings.warn(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored):\n{conversation}")
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            labels=target,
        )

    # noinspection PyMethodMayBeStatic
    def tk_conv_colon_two_eval(self, conv, tokenizer, **kwargs):
        assert len(conv.messages) >= 2
        # target = conv.messages[-1][-1]
        target = conv.get_prompt()

        conv.messages[-1][-1] = ""
        conversation = conv.get_prompt()
        input_ids = tokenizer([conversation, ], **kwargs).input_ids[0]

        target = tokenizer([target, ], add_special_tokens=False, **kwargs).input_ids[0]
        target[target == tokenizer.pad_token_id] = IGNORE_INDEX
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            labels=target,
        )


@FUNCTIONS.register_module()
class ShikraImageProcessor(BaseImageProcessFunc):
    def __call__(self, image: Image.Image, preprocessor: Dict[str, Any]) -> Dict[str, Any]:
        image_processor = preprocessor['image']

        if isinstance(image, (list, tuple)):
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            assert False, 'Shikra not support MultiImage'
        elif isinstance(image, PIL.Image.Image):
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            if hasattr(image_processor, 'crop_size'):
                crop_size = image_processor.crop_size
                height, width = crop_size['height'], crop_size['width']
            else:
                raise ValueError("got empty image. and don't know how to pad")
            image = torch.zeros(3, height, width)
        return {'image': image}
