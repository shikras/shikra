# client 端 api 调用案例---------------------------------
import os
import re
import base64
from io import BytesIO
from typing import Union

import torch
import requests
from PIL import Image
from torchvision.transforms import ToPILImage, PILToTensor
from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes


########################################
# helper
########################################

def pil_to_base64(pil_img):
    output_buffer = BytesIO()
    pil_img.save(output_buffer, format="PNG")
    byte_data = output_buffer.getvalue()
    encode_img = base64.b64encode(byte_data)
    return str(encode_img, encoding='utf-8')


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


def draw_bounding_boxes(
        image,
        boxes,
        **kwargs,
):
    if isinstance(image, Image.Image):
        image = PILToTensor()(image)
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    return _draw_bounding_boxes(image, boxes, **kwargs)


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


########################################
#
########################################

def query(image: Union[Image.Image, str], text: str, boxes_value: list, boxes_seq: list, server_url='http://127.0.0.1:12345/shikra'):
    if isinstance(image, str):
        image = Image.open(image)
    pload = {
        "img_base64": pil_to_base64(image),
        "text": text,
        "boxes_value": boxes_value,
        "boxes_seq": boxes_seq,
    }
    resp = requests.post(server_url, json=pload)
    if resp.status_code != 200:
        raise ValueError(resp.reason)
    ret = resp.json()
    return ret


def postprocess(text, image):
    if image is None:
        return text, None

    image = expand2square(image)

    colors = ['#ed7d31', '#5b9bd5', '#70ad47', '#7030a0', '#c00000', '#ffff00', "olive", "brown", "cyan"]
    pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')

    def extract_boxes(string):
        ret = []
        for bboxes_str in pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

    extract_pred = extract_boxes(text)
    boxes_to_draw = []
    color_to_draw = []
    for idx, boxes in enumerate(extract_pred):
        color = colors[idx % len(colors)]
        for box in boxes:
            boxes_to_draw.append(de_norm_box_xyxy(box, w=image.width, h=image.height))
            color_to_draw.append(color)
    if not boxes_to_draw:
        return text, None
    res = draw_bounding_boxes(image=image, boxes=boxes_to_draw, colors=color_to_draw, width=8)
    res = ToPILImage()(res)

    # post process text color
    location_text = text
    edit_text = list(text)
    bboxes_str = pat.findall(text)
    for idx in range(len(bboxes_str) - 1, -1, -1):
        color = colors[idx % len(colors)]
        boxes = bboxes_str[idx]
        span = location_text.rfind(boxes), location_text.rfind(boxes) + len(boxes)
        location_text = location_text[:span[0]]
        edit_text[span[0]:span[1]] = f'<span style="color:{color}; font-weight:bold;">{boxes}</span>'
    text = "".join(edit_text)
    return text, res


if __name__ == '__main__':
    server_url = 'http://127.0.0.1:12345' + "/shikra"


    def example1():
        image_path = os.path.join(os.path.dirname(__file__), 'assets/rec_bear.png')
        text = 'Can you point out a brown teddy bear with a blue bow in the image <image> and provide the coordinates of its location?'
        boxes_value = []
        boxes_seq = []

        response = query(image_path, text, boxes_value, boxes_seq, server_url)
        print(response)

        _, image = postprocess(response['response'], image=Image.open(image_path))
        print(_)
        if image is not None:
            image.show()


    def example2():
        image_path = os.path.join(os.path.dirname(__file__), 'assets/man.jpg')
        text = "What is the person<boxes> scared of?"
        boxes_value = [[148, 99, 576, 497]]
        boxes_seq = [[0]]

        response = query(image_path, text, boxes_value, boxes_seq, server_url)
        print(response)

        _, image = postprocess(response['response'], image=Image.open(image_path))
        print(_)
        if image is not None:
            image.show()


    example1()
    example2()
