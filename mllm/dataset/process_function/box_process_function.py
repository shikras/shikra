import re
import sys
import logging
import typing
from typing import List, Dict, Any, Tuple, Union

from ..utils.transform import norm_box_xyxy, norm_point_xyxy

from ..root import (
    FUNCTIONS,
    BaseTargetProcessFunc,
    BOXES_PLACEHOLDER,
    BOXES_PROCESSOR,
    POINTS_PLACEHOLDER,
)

from ...utils import smart_tokenizer_and_embedding_resize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]


@FUNCTIONS.register_module()
class BoxFormatProcess(BaseTargetProcessFunc):
    def __call__(self, raw_conv: List[Dict[str, Any]], target: Dict[str, Any], preprocessor: Dict[str, Any],
                 multimage_mode=False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        box_formatter = preprocessor['target']['boxes']

        if multimage_mode:
            target = typing.cast(list, target)
            outer_normalized_boxes = []
            for tgt in target:
                normalized_boxes = []
                if tgt is not None and 'boxes' in tgt:
                    for box in tgt['boxes']:
                        normalized_boxes.append(
                            norm_box_xyxy(box, w=tgt['width'], h=tgt['height'])
                        )
                outer_normalized_boxes.append(normalized_boxes)
            normalized_boxes = outer_normalized_boxes
            outer_normalized_points = []
            for tgt in target:
                normalized_points = []
                if tgt is not None and 'boxes' in tgt:
                    for box in tgt['boxes']:
                        normalized_points.append(
                            norm_box_xyxy(box, w=tgt['width'], h=tgt['height'])
                        )
                outer_normalized_points.append(normalized_points)
            normalized_points = outer_normalized_points
        else:
            # normalize target
            normalized_boxes = []
            if target is not None and 'boxes' in target:
                for box in target['boxes']:
                    normalized_boxes.append(
                        norm_box_xyxy(box, w=target['width'], h=target['height'])
                    )
            normalized_points = []
            if target is not None and 'points' in target:
                for point in target['points']:
                    normalized_points.append(
                        norm_point_xyxy(point, w=target['width'], h=target['height'])
                    )

        # convert bboxes_seq
        for sentence in raw_conv:
            words: str = sentence['value']
            boxes_seq: List[List[int]] = sentence.get('boxes_seq', None)
            if boxes_seq is not None:
                # map box seq
                boxes_seq: List[Boxes] = map_obj(normalized_boxes, boxes_seq)
                # reformat; replace <boxes> placeholder
                converted = box_formatter(words, boxes_seq)
                words = converted
            points_seq: List[List[int]] = sentence.get('points_seq', None)
            if points_seq is not None:
                # map point seq
                points_seq: List[Boxes] = map_obj(normalized_points, points_seq)
                # reformat; replace <points> placeholder
                converted = box_formatter.call_on_point(words, points_seq)
                words = converted
            if boxes_seq is not None or points_seq is not None:
                sentence['raw_value'] = sentence['value']
                sentence['value'] = words
        return raw_conv, target


def map_obj(boxes_value: List[List[float]], boxes_seq: List[List[int]]) -> List[List[List[float]]]:
    """
    >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
    >>> boxes_seq_ = [[3, 1], [2]]
    >>> var = map_obj(normalized_boxes, boxes_seq_)
    >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
    """
    try:
        ret = []
        for boxes in boxes_seq:
            boxes_ret = []
            for box_index in boxes:
                if isinstance(box_index, (list, tuple)):
                    boxes_ret.append(boxes_value[box_index[0]][box_index[1]])
                else:
                    boxes_ret.append(boxes_value[box_index])
            ret.append(boxes_ret)
        return ret
    except:
        raise SystemExit(f"error: map obj {boxes_value} {boxes_seq}")


class BoxFormatter:
    def __init__(self, bboxes_token=BOXES_PLACEHOLDER, points_token=POINTS_PLACEHOLDER):
        self.bboxes_token = bboxes_token
        self.points_token = points_token
        # normally the bboxes_token_pat is the same as bboxes_token if u not use some weird token
        self.bboxes_token_pat = re.compile(bboxes_token)
        self.points_token_pat = re.compile(points_token)

    def __call__(self, sentence: str, bboxes_seq: BoxesSeq) -> str:
        all_box = self.bboxes_token_pat.findall(sentence)
        assert len(all_box) == len(bboxes_seq), f"not match. sentence: {sentence}. boxes:{bboxes_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_box(bboxes) for bboxes in bboxes_seq]
        converted = sentence.replace(self.bboxes_token, '{}').format(*bboxes_strs)
        return converted

    def call_on_point(self, sentence: str, points_seq: BoxesSeq) -> str:
        all_box = self.points_token_pat.findall(sentence)
        assert len(all_box) == len(points_seq), f"not match. sentence: {sentence}. boxes:{points_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_point(bboxes) for bboxes in points_seq]
        converted = sentence.replace(self.points_token, '{}').format(*bboxes_strs)
        return converted

    def format_point(self, points) -> str:
        raise NotImplementedError

    def format_box(self, bboxes: Boxes) -> str:
        raise NotImplementedError

    def extract(self, string: str) -> List[Boxes]:
        raise NotImplementedError

    def extract_point(self, string: str) -> List[Boxes]:
        raise NotImplementedError


@BOXES_PROCESSOR.register_module()
class PlainBoxFormatter(BoxFormatter):

    def __init__(self, *args, precision=3, use_small_brackets=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.use_small_brackets = use_small_brackets

        small_brackets_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\)')
        small_brackets_point_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\)')

        middle_brackets_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')
        middle_brackets_point_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\]')

        self.pat = small_brackets_pat if use_small_brackets else middle_brackets_pat
        self.point_pat = small_brackets_point_pat if use_small_brackets else middle_brackets_point_pat

    def format_box(self, boxes: Boxes) -> str:
        box_strs = []
        for box in boxes:
            box_strs.append(','.join([f"{elem:.{self.precision}f}" for elem in box]))
        box_str = ';'.join(box_strs)
        if self.use_small_brackets:
            return "(" + box_str + ")"
        return "[" + box_str + "]"

    def format_point(self, points) -> str:
        return self.format_box(points)

    def extract(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

    def extract_point(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.point_pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret


@BOXES_PROCESSOR.register_module()
class TokenFormatter(BoxFormatter):

    def __init__(self, num_bins=1001):
        super().__init__()
        self.extract_box_pat = re.compile(r'<b_st><bin_\d*?>(?:<bin_\d*?>){3}(?:<b_sep><bin_\d*?>(?:<bin_\d*?>){3})*<b_ed>')
        self.extract_point_pat = re.compile(r'<p_st><bin_\d*?>(?:<bin_\d*?>){1}(?:<p_sep><bin_\d*?>(?:<bin_\d*?>){1})*<p_ed>')
        self.num_bins = num_bins
        self.use_sep = True
        self.use_begin_end = True

        self.box_begin = '<b_st>'
        self.box_sep = '<b_sep>'
        self.box_end = '<b_ed>'

        self.point_begin = '<p_st>'
        self.point_sep = '<p_sep>'
        self.point_end = '<p_ed>'

    def format_point(self, points) -> str:
        final_str = []
        for bbox in points:
            quant_x0 = "<bin_{}>".format(round((bbox[0] * (self.num_bins - 1))))
            quant_y0 = "<bin_{}>".format(round((bbox[1] * (self.num_bins - 1))))
            region_coord = "{} {}".format(quant_x0, quant_y0)
            final_str.append(region_coord)
        if self.use_sep:
            final_str = self.point_sep.join(final_str)
        else:
            final_str = ''.join(final_str)
        if self.use_begin_end:
            final_str = self.point_begin + final_str + self.point_end
        return final_str

    def format_box(self, bboxes: Boxes) -> str:
        final_str = []
        for bbox in bboxes:
            quant_x0 = "<bin_{}>".format(round((bbox[0] * (self.num_bins - 1))))
            quant_y0 = "<bin_{}>".format(round((bbox[1] * (self.num_bins - 1))))
            quant_x1 = "<bin_{}>".format(round((bbox[2] * (self.num_bins - 1))))
            quant_y1 = "<bin_{}>".format(round((bbox[3] * (self.num_bins - 1))))
            region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
            final_str.append(region_coord)
        if self.use_sep:
            final_str = self.box_sep.join(final_str)
        else:
            final_str = ''.join(final_str)
        if self.use_begin_end:
            final_str = self.box_begin + final_str + self.box_end
        return final_str

    def extract(self, string: str) -> List[Boxes]:
        ret = []
        for bboxes_str in self.extract_box_pat.findall(string.replace(" ", "")):
            bboxes = []
            bbox_strs = bboxes_str.replace(self.box_begin, "").replace(self.box_end, "").split(self.box_sep)
            for bbox_str in bbox_strs:
                elems = list(map(int, re.findall(r'<bin_(\d*?)>', bbox_str)))
                bbox = [elem / (self.num_bins - 1) for elem in elems]
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

    def extract_point(self, string: str) -> List[Boxes]:
        ret = []
        for bboxes_str in self.extract_point_pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace(self.point_begin, "").replace(self.point_end, "").split(self.point_sep)
            for bbox_str in bbox_strs:
                elems = list(map(int, re.findall(r'<bin_(\d*?)>', bbox_str)))
                bbox = [elem / (self.num_bins - 1) for elem in elems]
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

    def post_process_model_tokenizer(self, model, preprocessor, model_args, training_args):
        tokenizer = preprocessor['text']

        additional_special_tokens = [
            self.box_begin, self.box_sep, self.box_end,
            self.point_begin, self.point_sep, self.point_end,
        ]
        for i in range(self.num_bins):
            additional_special_tokens.append(f'<bin_{i}>')

        smart_tokenizer_and_embedding_resize(
            {'additional_special_tokens': additional_special_tokens},
            tokenizer,
            model,
        )
        return model, preprocessor


# FIXME: merge into load_pretrained
def prepare_target_processor(
        model,  # multimodal llm
        preprocessor: Dict[str, Any],
        model_args,
        training_args,
):
    if not hasattr(model_args, 'target_processor'):
        return model, preprocessor

    target_processor = {}
    if 'boxes' in model_args['target_processor']:
        boxes_cfg = model_args['target_processor']['boxes']
        boxes_processor = BOXES_PROCESSOR.build(boxes_cfg)
        target_processor['boxes'] = boxes_processor
        if hasattr(boxes_processor, "post_process_model_tokenizer"):
            model, preprocessor = boxes_processor.post_process_model_tokenizer(
                model, preprocessor, model_args, training_args,
            )
    preprocessor['target'] = target_processor
    return model, preprocessor
