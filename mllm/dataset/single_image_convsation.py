import warnings
from functools import partial
from typing import Dict, Any, Callable, List, Optional, Tuple, Type

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainingArguments

from .root import IMAGE_PLACEHOLDER, BOXES_PLACEHOLDER
from ..conversation import Conversation, get_conv_template
from ..utils import post_process_generate_ids


class SingleImageConvDatasetMixin:

    def __init__(
            self,
            *args,
            preprocessor: Dict[str, Any],
            process_func: Dict[str, Any],
            conv_template: Callable[[], Conversation] = partial(get_conv_template, name='vicuna_v1.1'),
            mode='train',
            tokenize_kwargs: dict = None,
            training_args: TrainingArguments = None,
            transforms: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert mode in ['train', 'validation', 'test']

        self.preprocessor = preprocessor
        self.process_func = process_func
        self.conv_template = conv_template
        self.mode = mode
        self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
        self.training_args = training_args
        self.transforms = transforms

    def __getitem__(self, index, debug_mode=False, return_conv=False) -> Dict[str, Any]:
        # getitem
        item = self.get_raw_item(index)
        image: Image.Image = item.get('image', None)
        target: Dict[str, Any] = item.get('target', None)
        raw_conv: List[Dict[str, Any]] = item['conversations']

        # transform
        assert isinstance(image, list) == isinstance(target, list)
        multimage_mode = isinstance(image, list)
        if isinstance(image, list):
            # TODO: validate raw item
            transformed_image, transformed_target = [], []
            for img, tgt in zip(image, target):
                if self.transforms is not None and image is not None:
                    img, tgt = self.transforms(img, tgt)
                if tgt is not None:
                    tgt['width'], tgt['height'] = img.width, img.height
                transformed_image.append(img)
                transformed_target.append(tgt)
            image, target = transformed_image, transformed_target
        else:
            self.validate_raw_item(item)  # only validate for single image.
            if self.transforms is not None and image is not None:
                image, target = self.transforms(image, target)
            has_image = 'image' in item and bool(item['image'])
            has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
            if has_target and has_image:
                target['width'], target['height'] = image.width, image.height

        # preprocess
        raw_conv = self.process_conv(raw_conv)
        raw_conv, image = self.process_conv_multimage(raw_conv, image)
        raw_conv, _ = self.process_target(raw_conv, target, multimage_mode=multimage_mode)
        conv = self.build_conv(raw_conv)
        if return_conv:
            # noinspection PyTypeChecker
            return conv
        text_dict = self.process_text(conv)
        image_dict = self.process_image(image)

        # return
        ret_dict = {}
        ret_dict.update(text_dict)
        ret_dict.update(image_dict)
        self._print_sample(ret_dict, raw_conv, conv)
        if debug_mode:
            return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': conv, 'image': image}
        return ret_dict

    def __len__(self):
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def process_conv_multimage(self, raw_conv, image):
        # re-sort multi image
        if image is None:
            return raw_conv, image
        if not isinstance(image, (list, tuple)):
            return raw_conv, image
        image_seqs = []
        for conv in raw_conv:
            image_seqs.extend(conv['image_seq'] if 'image_seq' in conv else [])
        images = []
        for idx in image_seqs:
            images.append(image[idx])
        return raw_conv, images

    def get_raw_item(self, index) -> Dict[str, Any]:
        """
        return item format like this.
        item = {
            'image': # PIL.Image.Image,
            'target': {
                # xmin, ymin, xmax, ymax
                'boxes': [
                    [10, 10, 256, 265],  # dog1
                    [24, 18, 378, 768],  # dog2
                    [100, 310, 670, 653],  # man
                    [278, 320, 809, 673],  # rope
                ],
            }

            "conversations": [
                {
                    'from': 'human',
                    'value': 'What is the relation between the two dogs <boxes> and the man <boxes> in the image <image> ?',
                    'boxes_seq': [[0, 1], [2], ],
                },
                {
                    'from': 'gpt',
                    'value': 'a rope <boxes> is connecting the left dog <boxes> with the man <boxes>. '
                             'So the man <boxes> is walking the dog <boxes>.'
                            'And the man <boxes> has no relationship with the right dog <boxes>',
                    'boxes_seq': [[3], [0], [2], [2], [0], [2], [1]],
                }
            ]
        }
        # placeholder: <image> <boxes>
        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def validate_raw_item(self, item):
        has_image = 'image' in item and bool(item['image'])
        has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
        has_target_boxes = 'boxes' in item['target'] if has_target else False
        raw_conv: List[Dict[str, Any]] = item['conversations']

        # check image
        human_input_has_image_placeholder = any(
            sentence['from'] == 'human' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        if human_input_has_image_placeholder:
            assert has_image
        if has_image and (not human_input_has_image_placeholder):
            warnings.warn(f'item has image but the question has no image placeholder.\n{item}')
        gpt_input_has_image_placeholder = any(
            sentence['from'] == 'gpt' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        assert not gpt_input_has_image_placeholder

        # check target
        has_boxes_placeholder = any(
            BOXES_PLACEHOLDER in sentence['value'] for sentence in raw_conv
        )
        if has_boxes_placeholder:
            assert has_target_boxes
        # not check box placeholder num this will be checked in format process

    def build_conv(self, source: List[Dict[str, Any]]) -> Conversation:
        conv = self.conv_template()
        role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
        assert len(source) > 0
        assert source[0]['from'] == 'human'
        for sentence in source:
            role = role_map[sentence['from']]
            conv.append_message(role, sentence['value'])
        return conv

    def process_conv(self, raw_conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        some utils preprocess for raw_conv.
            e.g. replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
        """
        return self.process_func['conv'](raw_conv, self.preprocessor, self.conv_template)

    def process_target(self, raw_conv: List[Dict[str, Any]], target: Dict[str, Any], multimage_mode=False) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
        """
        convert target placeholder to actual information in raw_conv.
            e.g. normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
        """
        return self.process_func['target'](raw_conv, target, self.preprocessor, multimage_mode=multimage_mode)

    def process_text(self, conv: Conversation) -> Dict[str, Any]:
        """
        convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.
            self.tokenize_kwargs control something like padding/truncation behavior.
        """
        return self.process_func['text'](conv, self.preprocessor, self.mode, **self.tokenize_kwargs)

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        convert Image.Image object to torch.Tensor
        """
        return self.process_func['image'](image, self.preprocessor)

    def _print_sample(self, ret_dict, raw_conv, conv):
        if not hasattr(self, '_printed_sample'):
            self._printed_sample = True
            post_processed_labels = post_process_generate_ids(self.preprocessor['text'], ret_dict['labels'])
            print(f"=================== {self.mode} sample ===================", flush=True)
            print(f"        input_ids: {self.preprocessor['text'].convert_ids_to_tokens(ret_dict['input_ids'])}")
            print(f"           labels: {self.preprocessor['text'].convert_ids_to_tokens(post_processed_labels)}")
            print(f"decoded input_ids: {self.preprocessor['text'].decode(ret_dict['input_ids'])}")
            print(f"decoded    labels: {self.preprocessor['text'].decode(post_processed_labels)}")
            if 'image' in ret_dict and ret_dict['image'] is not None:
                image = ret_dict['image']
                if isinstance(image, torch.Tensor):
                    print(f"            image: {image.shape}")
                elif isinstance(image, dict):
                    print(f"            image: {image.keys()}")
                elif isinstance(image, list) and len(image) > 0:
                    print(f"            image: {len(image)}, {type(image[0])}")
                else:
                    print(f"            image: {type(image)}")
            print("====================================================", flush=True)
            try:
                if self.training_args is not None:
                    _save_obj = {
                        'ret_dict': ret_dict,
                        'raw_conv': raw_conv,
                        'conv': conv.get_prompt(),
                    }
                    from pathlib import Path
                    output_dir = Path(self.training_args.output_dir)
                    output_dir.mkdir(exist_ok=True, parents=True)
                    _local_rank = self.training_args.local_rank
                    _word_size = self.training_args.world_size
                    _file_path = str(output_dir / f'sample_check_{self.mode}_{_local_rank}_{_word_size}.pt')
                    print(f'saving some sample to {_file_path} for check.')
                    torch.save(_save_obj, _file_path)
            except Exception as e:
                warnings.warn(f'try to save samples but get exception: {e.args}. ignored.')


class SingleImageConvDataset(SingleImageConvDatasetMixin, Dataset):
    _repr_indent = 4

    def __init__(self, *args, dataset_generator: Type[Dataset], **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_generator = dataset_generator
        self.dataset = None

    def initialize_if_needed(self):
        """
        lazy initialize for big in-memory python object due to python 'copy-on-read' behavior
        when num_worker > 0. refer: https://github.com/pytorch/pytorch/issues/13246
        """
        if self.dataset is None:
            # warnings.warn("it's highly recommended that set persistent_workers=True, "
            #               "otherwise this initialize code will run in every epoch beginning."
            #               "(ignore me if set)")
            self.dataset = self.dataset_generator()

    def __len__(self):
        self.initialize_if_needed()
        return len(self.dataset)

    def get_raw_item(self, index) -> Dict[str, Any]:
        self.initialize_if_needed()
        return self.dataset[index]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        body += self.dataset.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


__all__ = ['SingleImageConvDatasetMixin', 'SingleImageConvDataset']
