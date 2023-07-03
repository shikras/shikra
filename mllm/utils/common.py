import copy
from typing import List, Union, Dict

import PIL.Image
import torch
import numpy as np
import torchvision.transforms.functional as F
import transformers
from matplotlib import pyplot as plt

from transformers import PreTrainedTokenizer


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param))


def post_process_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor):
    ids = copy.deepcopy(ids)  # do not modify origin preds and targets
    ids[ids < 0] = tokenizer.pad_token_id
    return ids


def decode_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor) -> Union[List[str], str]:
    assert ids.ndim in [1, 2]
    only_one_sentence = ids.ndim == 1
    if only_one_sentence:
        ids = ids.unsqueeze(0)
    ids = post_process_generate_ids(tokenizer, ids)
    res = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if only_one_sentence:
        return res[0]
    return res


def show(imgs: Union[torch.Tensor, List[Union[torch.Tensor, PIL.Image.Image]]]):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def draw_bounding_boxes(
        image: Union[torch.Tensor, PIL.Image.Image],
        boxes: Union[torch.Tensor, List, np.ndarray],
        **kwargs,
):
    if isinstance(image, PIL.Image.Image):
        from torchvision.transforms import PILToTensor
        image = PILToTensor()(image)
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes
    return _draw_bounding_boxes(image, boxes, **kwargs)


# https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087
def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
