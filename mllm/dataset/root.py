from typing import Dict, Any, List, Tuple

from PIL import Image
from mmengine import DATASETS, TRANSFORMS, METRICS, FUNCTIONS, Registry

from ..conversation import Conversation

IMAGE_PLACEHOLDER = '<image>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
OBJS_PLACEHOLDER = '<objs>'
QUESTION_PLACEHOLDER = '<question>'
POINTS_PLACEHOLDER = '<points>'
# processor
BOXES_PROCESSOR = Registry('Processor for Boxes')


# only for static type checking
class BaseConvProcessFunc:
    def __call__(
            self,
            raw_conv: List[Dict[str, Any]],
            preprocessor: Dict[str, Any],
            conv_template: Conversation,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class BaseTargetProcessFunc:
    def __call__(
            self,
            raw_conv: List[Dict[str, Any]],
            target: Dict[str, Any],
            preprocessor: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raise NotImplementedError


class BaseTextProcessFunc:
    def __call__(
            self,
            conv: Conversation,
            preprocessor: Dict[str, Any],
            mode: str,
            **tokenize_kwargs,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class BaseImageProcessFunc:
    def __call__(
            self,
            image: Image.Image,
            preprocessor: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


__all__ = [
    'IMAGE_PLACEHOLDER', 'BOXES_PLACEHOLDER', 'EXPR_PLACEHOLDER', 'OBJS_PLACEHOLDER', 'QUESTION_PLACEHOLDER', 'POINTS_PLACEHOLDER',
    'FUNCTIONS',
    'DATASETS',
    'TRANSFORMS',
    'METRICS',
    'BOXES_PROCESSOR',
    'BaseConvProcessFunc', 'BaseTargetProcessFunc', 'BaseTextProcessFunc', 'BaseImageProcessFunc',
]
