import os
import sys
import logging
import time
import argparse
import tempfile
from pathlib import Path
from typing import List, Any, Union

import torch
import numpy as np
import gradio as gr
from PIL import Image
from PIL import ImageDraw, ImageFont
from mmengine import Config
import transformers
from transformers import BitsAndBytesConfig

sys.path.append(str(Path(__file__).parent.parent.parent))

from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.utils import draw_bounding_boxes
from mllm.models.builder.build_shikra import load_pretrained_shikra

log_level = logging.DEBUG
transformers.logging.set_verbosity(log_level)
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

TEMP_FILE_DIR = Path(__file__).parent / 'temp'
TEMP_FILE_DIR.mkdir(parents=True, exist_ok=True)

#########################################
# mllm model init
#########################################
parser = argparse.ArgumentParser("Shikra Web Demo")
parser.add_argument('--model_path', required=True)
parser.add_argument('--load_in_8bit', action='store_true')
parser.add_argument('--server_name', default=None)
parser.add_argument('--server_port', type=int, default=None)

args = parser.parse_args()
print(args)

model_name_or_path = args.model_path

model_args = Config(dict(
    type='shikra',
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=model_name_or_path,
    vision_tower=r'openai/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=None,

    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,

    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='ShikraConvProcess'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='ShikraTextProcess'),
        image=dict(type='ShikraImageProcessor'),
    ),

    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=None),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
))
training_args = Config(dict(
    bf16=False,
    fp16=True,
    device='cuda',
    fsdp=None,
))

if args.load_in_8bit:
    quantization_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
        )
    )
else:
    quantization_kwargs = dict()

model, preprocessor = load_pretrained_shikra(model_args, training_args, **quantization_kwargs)
if not getattr(model, 'is_quantized', False):
    model.to(dtype=torch.float16, device=torch.device('cuda'))
if not getattr(model.model.vision_tower[0], 'is_quantized', False):
    model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device('cuda'))
print(f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
print(f"vision device: {model.model.vision_tower[0].device}, is_quantized: {getattr(model.model.vision_tower[0], 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")

preprocessor['target'] = {'boxes': PlainBoxFormatter()}
tokenizer = preprocessor['text']


#########################################
# demo utils
#########################################

def parse_text(text):
    text = text.replace("<image>", "&lt;image&gt;")
    return text


def setup_gradio_warning(level=1):
    """
    level            0       1           2        3
    level          IGNORE   Weak       Strong    Error
    has Warning      _foo   Warning    Warning   Error
    no Warning       _foo    _foo      Error     Error
    """

    def _dummy_func(*args, **kwargs):
        pass

    def _raise_error(*args, **kwargs):
        raise gr.Error(*args, **kwargs)

    assert level in [0, 1, 2, 3]
    if level >= 3:
        return _raise_error
    if level <= 0:
        return _dummy_func
    if hasattr(gr, 'Warning'):
        return gr.Warning
    if level == 1:
        return _dummy_func
    return _raise_error


grWarning = setup_gradio_warning()


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


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


def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box


def resize_pil_img(pil_img: Image.Image, *, w, h):
    old_height, old_width = pil_img.height, pil_img.width
    new_height, new_width = (h, w)
    if (new_height, new_width) == (old_height, old_width):
        return pil_img
    return pil_img.resize((new_width, new_height))


def resize_box_xyxy(boxes, *, w, h, ow, oh):
    old_height, old_width = (oh, ow)
    new_height, new_width = (h, w)
    if (new_height, new_width) == (old_height, old_width):
        return boxes
    w_ratio = new_width / old_width
    h_ratio = new_height / old_height
    out_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = x1 * w_ratio
        x2 = x2 * w_ratio
        y1 = y1 * h_ratio
        y2 = y2 * h_ratio
        nb = (x1, y1, x2, y2)
        out_boxes.append(nb)
    return out_boxes


# use mask to simulate box
# copy from https://github.com/gligen/GLIGEN/blob/master/demo/app.py
class ImageMask(gr.components.Image):
    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)


def binarize(x):
    return (x != 0).astype('uint8') * 255


class ImageBoxState:
    def __init__(self, draw_size: Union[int, float, tuple, list] = 512):
        if isinstance(draw_size, (float, int)):
            draw_size = (draw_size, draw_size)
        assert len(draw_size) == 2
        self.size = draw_size
        self.height, self.width = self.size[0], self.size[1]
        self.reset_state()

    # noinspection PyAttributeOutsideInit
    def reset_state(self):
        self.image = None
        self.boxes = []
        self.masks = []

    # noinspection PyAttributeOutsideInit
    def reset_masks(self):
        self.boxes = []
        self.masks = []

    # noinspection PyAttributeOutsideInit
    def update_image(self, image):
        if image != self.image:
            self.reset_state()
            self.image = image

    def update_mask(self, mask):
        if len(self.masks) == 0:
            last_mask = np.zeros_like(mask)
        else:
            last_mask = self.masks[-1]

        if type(mask) == np.ndarray and mask.size > 1:
            diff_mask = mask - last_mask
        else:
            diff_mask = np.zeros([])

        if diff_mask.sum() > 0:
            # noinspection PyArgumentList
            x1x2 = np.where(diff_mask.max(0) != 0)[0]
            # noinspection PyArgumentList
            y1y2 = np.where(diff_mask.max(1) != 0)[0]
            y1, y2 = y1y2.min(), y1y2.max()
            x1, x2 = x1x2.min(), x1x2.max()
            if (x2 - x1 > 5) and (y2 - y1 > 5):
                self.masks.append(mask.copy())
                self.boxes.append(tuple(map(int, (x1, y1, x2, y2))))

    def update_box(self, box):
        x1, y1, x2, y2 = box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.boxes.append(tuple(map(int, (x1, y1, x2, y2))))

    def to_model(self):
        if self.image is None:
            return {}
        image = expand2square(self.image)
        boxes = [box_xyxy_expand2square(box, w=self.image.width, h=self.image.height) for box in self.boxes]
        return {'image': image, 'boxes': boxes}

    def draw_boxes(self):
        assert self.image is not None
        grounding_texts = [f'{bid}' for bid in range(len(self.boxes))]
        image = expand2square(self.image)
        boxes = [box_xyxy_expand2square(box, w=self.image.width, h=self.image.height) for box in self.boxes]

        image_to_draw = resize_pil_img(image, w=self.width, h=self.height)
        boxes_to_draw = resize_box_xyxy(boxes, w=self.width, h=self.height, ow=image.width, oh=image.height)

        def _draw(img, _boxes: List[Any], texts: List[str]):
            assert img is not None
            colors = ["red", "blue", "green", "olive", "orange", "brown", "cyan", "purple"]
            _img_draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'assets/DejaVuSansMono.ttf'), size=18)
            for bid, box in enumerate(_boxes):
                _img_draw.rectangle((box[0], box[1], box[2], box[3]), outline=colors[bid % len(colors)], width=4)
                anno_text = texts[bid]
                _img_draw.rectangle((box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]),
                                    outline=colors[bid % len(colors)], fill=colors[bid % len(colors)], width=4)
                _img_draw.text((box[0] + int(font.size * 0.2), box[3] - int(font.size * 1.2)), anno_text, font=font, fill=(255, 255, 255))
            return img

        out_draw = _draw(image_to_draw, boxes_to_draw, grounding_texts)
        return out_draw


def add_submit_temp_image(state, temp_image_path):
    if '_submit_temp_images' not in state:
        state['_submit_temp_images'] = []
    state['_submit_temp_images'].append(temp_image_path)
    return state


def clear_submit_temp_image(state):
    if '_submit_temp_images' in state:
        for path in state['_submit_temp_images']:
            os.remove(path)
        del state['_submit_temp_images']
    return state


if __name__ == '__main__':
    with gr.Blocks() as demo:
        logo_file_url = f"file={os.path.join(os.path.dirname(__file__), 'assets/logo.png')}"
        gr.HTML(
            f"""

<p align="center"><img src="{logo_file_url}" alt="Logo" width="130"></p>
<h1 align="center"><font color="#966661">Shikra</font>: Unleashing Multimodal LLM’s Referential Dialogue Magic</h1>
<p align="center">
    <a href='https://github.com/shikras/shikra' target='_blank'>[Project]</a>
    <a href='http://arxiv.org/abs/2306.15195' target='_blank'>[Paper]</a>
</p>
<p>
    <font color="#966661"><strong>Shikra</strong></font>, an MLLM designed to kick off <strong>referential dialogue</strong> by excelling in spatial coordinate inputs/outputs in natural language, <strong>without</strong> additional vocabularies, position encoders, pre-/post-detection, or external plug-in models.
</p>
<h2>User Manual</h2>
<ul>
<li><p><strong>Step 1.</strong> Upload an image</p>
</li>
<li><p><strong>Step 2.</strong> Select Question Format in <code>Task Template</code>.  Task template and user input (if exists) will be assembled into final inputs to the model.</p>
<ul>
<li><strong>SpotCap</strong>: Ask the model to generate a <strong>grounded caption</strong>.</li>
<li><strong>GCoT</strong>: Ask the model to answer the question and provide a <strong>Grounding-CoT</strong>, which is a step-by-step reasoning with explicit grounding information.</li>
<li><strong>Cap</strong>: Ask the model to generate a <strong>short caption</strong>.</li>
<li><strong>VQA</strong>: Ask the model to answer the question <strong>directly</strong>.</li>
<li><strong>REC</strong>: <strong>Referring Expression Comprehension</strong>. Ask the model to output the bounding box of <code>&lt;expr&gt;</code>. </li>
<li><strong>REG</strong>: <strong>Referring Expression Generation</strong>. Ask the model to generate a distinguishable description for RoI.</li>
<li><strong>Advanced</strong>: Use no predefined template. You can take full control of inputs.</li>

</ul>
</li>

<li><p><strong>Step 3.</strong> Ask Question. Use &lt;boxes&gt; placeholder if input has bounding box.</p>
</li>

</ul>
<p>The following step are needed <strong>only</strong> when input has bounding box.</p>
<ul>
<li><p><strong>Step 4.</strong> Draw Bounding Box in <code>Sketch Pad</code>.</p>
<p>Each bbox has a unique index, which will show at the corner of the bbox in <code>Parsed Sketch Pad</code>. </p>
</li>
<li><p><strong>Step 5.</strong> Assign the bbox index in <code>Boexs Seq</code> for each &lt;boxes&gt; placeholder. <code>Boexs Seq</code> <strong>take a 2-d list as input, each sub-list will replace the &lt;boxes&gt; placeholder in order.</strong></p>
</li>
</ul>
"""
        )

        with gr.Row():
            with gr.Column():
                gr.HTML(
                    """
                    <h2>Video example</h2>
                    <p>a video example demonstrate how to input with boxes</p>
                    """
                )
                video_file_url = os.path.join(os.path.dirname(__file__), f"assets/petal_20230711_153216_Compressed.mp4")
                gr.Video(value=video_file_url, interactive=False, width=600)
            with gr.Column():
                boxes_seq_usage_file_url = f'file={os.path.join(os.path.dirname(__file__), f"assets/boxes_seq_explanation.jpg")}'
                gr.HTML(
                    f"""
<h2>Boxes Seq Usage Explanation</h2>
<p>the [0,2] boxes will replace the first &lt;boxes&gt; placeholder. the [1] boxes will replace the second &lt;boxes&gt; placeholder.</p>
<p><img src="{boxes_seq_usage_file_url}"></p>
"""
                )

        gr.HTML(
            """
            <h2>Demo</h2>
            """
        )
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot()
                with gr.Accordion("Parameters", open=False):
                    with gr.Row():
                        do_sample = gr.Checkbox(value=False, label='do sampling', interactive=True)
                        max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="max length", interactive=True)
                        top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                        temperature = gr.Slider(0, 10, value=0.75, step=0.01, label="Temperature", interactive=True)
            with gr.Column():
                with gr.Row(variant='compact'):
                    sketch_pad = ImageMask(label="Sketch Pad", elem_id="img2img_image")
                    out_imagebox = gr.Image(label="Parsed Sketch Pad")
                with gr.Column():
                    radio = gr.Radio(
                        ["SpotCap", "GCoT", "Cap", "VQA", "REC", "REG", "Advanced"], label="Task Template", value='SpotCap',
                    )
                    with gr.Group():
                        template = gr.Textbox(label='Template', show_label=True, lines=1, interactive=False,
                                              value='Provide a comprehensive description of the image <image> and specify the positions of any mentioned objects in square brackets.')
                        user_input = gr.Textbox(label='<question>', show_label=True, placeholder="Input...", lines=3,
                                                value=None, visible=False, interactive=False)
                        boxes_seq = gr.Textbox(label='Boxes Seq', show_label=False, placeholder="Boxes Seq...", lines=1,
                                               value=None, visible=False, interactive=False)
                with gr.Row():
                    reset_all = gr.Button('Reset All')
                    reset_chat = gr.Button('Reset Chat')
                    reset_boxes = gr.Button('Reset Boxes')
                    submitBtn = gr.Button('Run')


        ##############################################
        #  reset state
        ##############################################

        def reset_state_func():
            ret = {
                'ibs': ImageBoxState(),
                'ds': prepare_interactive(model_args, preprocessor),
            }
            return ret


        state = gr.State(reset_state_func)
        example_image_boxes = gr.State(None)


        ##############################################
        #  reset dialogue
        ##############################################

        def reset_all_func(state):
            # clear_submit_temp_image(state)
            new_state = reset_state_func()
            boxes_seq = '[[0]]' if radio in ['REG', 'GC'] else None
            return [new_state, None, None, None, boxes_seq, None]


        reset_all.click(
            fn=reset_all_func,
            inputs=[state],
            outputs=[state, sketch_pad, out_imagebox, user_input, boxes_seq, chatbot],
        )


        def reset_chat_func_step1(state, radio):
            state['ibs'].reset_masks()
            new_state = reset_state_func()
            new_state['_reset_boxes_func_image'] = state['ibs'].image
            boxes_seq = '[[0]]' if radio in ['REG', 'GC'] else None
            return [new_state, None, None, None, boxes_seq, None]


        def reset_chat_func_step2(state):
            image = state['_reset_boxes_func_image']
            del state['_reset_boxes_func_image']
            return state, gr.update(value=image)


        reset_chat.click(
            fn=reset_chat_func_step1,
            inputs=[state, radio],
            outputs=[state, sketch_pad, out_imagebox, user_input, boxes_seq, chatbot],
        ).then(
            fn=reset_chat_func_step2,
            inputs=[state],
            outputs=[state, sketch_pad],
        )


        ##############################################
        #  reset boxes
        ##############################################

        def reset_boxes_func_step1(state):
            state['_reset_boxes_func_image'] = state['ibs'].image
            state['ibs'].reset_masks()
            return state, None


        def reset_boxes_func_step2(state):
            image = state['_reset_boxes_func_image']
            del state['_reset_boxes_func_image']
            return state, gr.update(value=image)


        # reset boxes
        reset_boxes.click(
            fn=reset_boxes_func_step1,
            inputs=[state],
            outputs=[state, sketch_pad],
        ).then(
            fn=reset_boxes_func_step2,
            inputs=[state],
            outputs=[state, sketch_pad],
        )


        ##############################################
        #  examples
        ##############################################

        def parese_example(image, boxes):
            state = reset_state_func()
            image = Image.open(image)
            state['ibs'].update_image(image)
            for box in boxes:
                state['ibs'].update_box(box)
            image = state['ibs'].draw_boxes()

            _, path = tempfile.mkstemp(suffix='.jpg', dir=TEMP_FILE_DIR)
            image.save(path)
            return path, state


        with gr.Column(visible=True) as example_SpotCap:
            _examples_cap_raw = [
                os.path.join(os.path.dirname(__file__), 'assets/proposal.jpg'),
                os.path.join(os.path.dirname(__file__), 'assets/water_question.jpg'),
                os.path.join(os.path.dirname(__file__), 'assets/fishing.jpg'),
                os.path.join(os.path.dirname(__file__), 'assets/ball.jpg'),

                os.path.join(os.path.dirname(__file__), 'assets/banana_phone.png'),
                os.path.join(os.path.dirname(__file__), "assets/airplane.jpg"),
                os.path.join(os.path.dirname(__file__), 'assets/baseball.png'),
            ]
            _examples_cap_parsed = [[item, []] for item in _examples_cap_raw]
            gr.Examples(
                examples=_examples_cap_parsed,
                inputs=[sketch_pad, example_image_boxes],
            )

        with gr.Column(visible=False) as example_vqabox:
            _examples_vqabox_parsed = [
                [
                    os.path.join(os.path.dirname(__file__), 'assets/proposal.jpg'),
                    'How is the person in the picture feeling<boxes>?',
                    '[[0]]',
                    [[785, 108, 1063, 844]],
                ],
                [
                    os.path.join(os.path.dirname(__file__), 'assets/woman_door.jpg'),
                    "Which one is the woman's reflection in the mirror?<boxes>",
                    '[[0,1]]',
                    [(770, 138, 1024, 752), (469, 146, 732, 744)],
                ],
                [
                    os.path.join(os.path.dirname(__file__), 'assets/man.jpg'),
                    "What is the person<boxes> scared of?",
                    '[[0]]',
                    [(148, 99, 576, 497)],
                ],
                [
                    os.path.join(os.path.dirname(__file__), "assets/giraffes.jpg"),
                    "How many animals in the image?",
                    "",
                    [],
                ],
                [
                    os.path.join(os.path.dirname(__file__), "assets/dog_selfcontrol.jpg"),
                    "Is this dog on a lead held by someone able to control it?",
                    "",
                    [],
                ],
                [
                    os.path.join(os.path.dirname(__file__), 'assets/wet_paint1.jpg'),
                    'What does the board say?',
                    '',
                    [],
                ],
                [
                    os.path.join(os.path.dirname(__file__), 'assets/g2.jpg'),
                    "What is unusual about the image?",
                    '',
                    [],
                ],
            ]

            gr.Examples(
                examples=_examples_vqabox_parsed,
                inputs=[sketch_pad, user_input, boxes_seq, example_image_boxes],
            )

        with gr.Column(visible=False) as example_vqa:
            _examples_vqa_parsed = [
                [
                    os.path.join(os.path.dirname(__file__), 'assets/food-1898194_640.jpg'),
                    "QUESTION: Which of the following is meat?\nOPTION:\n(A) <boxes>\n(B) <boxes>\n(C) <boxes>\n(D) <boxes>",
                    '[[0],[1],[2],[3]]',
                    [[20, 216, 70, 343], [8, 3, 187, 127], [332, 386, 424, 494], [158, 518, 330, 605]],
                ],
                [
                    os.path.join(os.path.dirname(__file__), 'assets/potato.jpg'),
                    "What color is this<boxes>?",
                    '[[0]]',
                    [[75, 408, 481, 802]],
                ],
                [
                    os.path.join(os.path.dirname(__file__), 'assets/potato.jpg'),
                    "What color is this<boxes>?",
                    '[[0]]',
                    [[147, 274, 266, 437]],
                ],
                [
                    os.path.join(os.path.dirname(__file__), 'assets/staircase-274614_640.jpg'),
                    "Is this a sea snail?",
                    '',
                    [],
                ],
                [
                    os.path.join(os.path.dirname(__file__), 'assets/staircase-274614_640.jpg'),
                    "Is this shape like a sea snail?",
                    '',
                    [],
                ],
            ]
            gr.Examples(
                examples=_examples_vqa_parsed,
                inputs=[sketch_pad, user_input, boxes_seq, example_image_boxes],
            )

        with gr.Column(visible=False) as example_rec:
            gr.Examples(
                examples=[
                    [
                        os.path.join(os.path.dirname(__file__), "assets/rec_bear.png"),
                        "a brown teddy bear with a blue bow",
                        [],
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/bear-792466_1280.jpg"),
                        "the teddy bear lay on the sofa edge",
                        [],
                    ],
                ],
                inputs=[sketch_pad, user_input, example_image_boxes],
            )

        with gr.Column(visible=False) as example_reg:
            gr.Examples(
                examples=[
                    [
                        os.path.join(os.path.dirname(__file__), "assets/fruits.jpg"),
                        "[[0]]",
                        [[833, 527, 646, 315]],
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/bearhat.png"),
                        "[[0]]",
                        [[48, 49, 216, 152]],
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/oven.jpg"),
                        "[[0]]",
                        [[1267, 314, 1383, 458]],
                    ],
                ],
                inputs=[sketch_pad, boxes_seq, example_image_boxes],
            )

        with gr.Column(visible=False) as example_adv:
            gr.Examples(
                examples=[
                    [

                    ],
                ],
                inputs=[sketch_pad, user_input, boxes_seq, example_image_boxes],
            )


        ##############################################
        #  task template select
        ##############################################

        def change_textbox(choice):
            task_template = {
                "SpotCap": "Provide a comprehensive description of the image <image> and specify the positions of any mentioned objects in square brackets.",
                "Cap": "Summarize the content of the photo <image>.",
                "GCoT": "With the help of the image <image>, can you clarify my question '<question>'? Also, explain the reasoning behind your answer, and don't forget to label the bounding boxes of the involved objects using square brackets.",
                "VQA": "For this image <image>, I want a simple and direct answer to my question: <question>",
                "REC": "Can you point out <expr> in the image <image> and provide the coordinates of its location?",
                "REG": "For the given image <image>, can you provide a unique description of the area <boxes>?",
                "GC": "Can you give me a description of the region <boxes> in image <image>?",
                "Advanced": "<question>",
            }
            if choice in ['Advanced']:
                template_update = gr.update(value=task_template[choice], visible=False)
            else:
                template_update = gr.update(value=task_template[choice], visible=True)

            if choice in ['SpotCap', 'Cap']:
                input_update = gr.update(value=None, visible=False, interactive=False)
                boxes_seq_update = gr.update(show_label=False, value=None, visible=False, interactive=False)
            elif choice in ['GCoT', 'VQA']:
                input_update = gr.update(label='<question>', value=None, visible=True, interactive=True)
                boxes_seq_update = gr.update(show_label=False, value=None, visible=True, interactive=True)
            elif choice in ['Advanced']:
                input_update = gr.update(label='Input', value=None, visible=True, interactive=True)
                boxes_seq_update = gr.update(show_label=False, value=None, visible=True, interactive=True)
            elif choice in ['REC']:
                input_update = gr.update(label='<expr>', value=None, visible=True, interactive=True)
                boxes_seq_update = gr.update(show_label=False, value=None, visible=False, interactive=False)
            elif choice in ['REG', 'GC']:
                input_update = gr.update(value=None, visible=False, interactive=False)
                boxes_seq_update = gr.update(show_label=True, value='[[0]]', visible=True, interactive=True)
            else:
                raise gr.Error("What is this?!")

            ret = [
                template_update,
                input_update,
                boxes_seq_update,
                gr.update(visible=True) if choice in ['SpotCap', 'Cap'] else gr.update(visible=False),
                gr.update(visible=True) if choice in ['GCoT'] else gr.update(visible=False),
                gr.update(visible=True) if choice in ['VQA'] else gr.update(visible=False),
                gr.update(visible=True) if choice in ['REC'] else gr.update(visible=False),
                gr.update(visible=True) if choice in ['REG', 'GC'] else gr.update(visible=False),
                gr.update(visible=True) if choice in ['Advanced'] else gr.update(visible=False),
            ]
            return ret


        radio.change(
            fn=change_textbox,
            inputs=radio,
            outputs=[template, user_input, boxes_seq, example_SpotCap, example_vqabox, example_vqa, example_rec, example_reg, example_adv],
        )


        ##############################################
        #  draw
        ##############################################

        def draw(sketch_pad: dict, state: dict, example_image_boxes):
            if example_image_boxes is None:
                image = sketch_pad['image']
                image = Image.fromarray(image)
                mask = sketch_pad['mask'][..., 0] if sketch_pad['mask'].ndim == 3 else sketch_pad['mask']
                mask = binarize(mask)
                ibs: ImageBoxState = state['ibs']
                ibs.update_image(image)
                ibs.update_mask(mask)
                out_draw = ibs.draw_boxes()
                ret = [out_draw, state, None, gr.update()]
                return ret
            else:
                image = sketch_pad['image']
                image = Image.fromarray(image)

                state = reset_state_func()
                ibs: ImageBoxState = state['ibs']
                ibs.update_image(image)
                for box in example_image_boxes:
                    ibs.update_box(box)
                out_draw = ibs.draw_boxes()
                ret = [out_draw, state, None, []]
                return ret


        sketch_pad.edit(
            fn=draw,
            inputs=[sketch_pad, state, example_image_boxes],
            outputs=[out_imagebox, state, example_image_boxes, chatbot],
            queue=False,
        )


        ##############################################
        #  submit boxes
        ##############################################

        def submit_step1(state, template, raw_user_input, boxes_seq, chatbot, do_sample, max_length, top_p, temperature):
            if '<expr>' in template or '<question>' in template:
                if not bool(raw_user_input):
                    raise gr.Error("say sth bro.")
            if '<expr>' in template:
                user_input = template.replace("<expr>", raw_user_input)
            elif '<question>' in template:
                user_input = template.replace("<question>", raw_user_input)
            else:
                user_input = template

            def parse_boxes_seq(boxes_seq_str) -> List[List[int]]:
                if not bool(boxes_seq_str):
                    return []
                import ast
                # validate
                try:
                    parsed = ast.literal_eval(boxes_seq_str)
                    assert isinstance(parsed, (tuple, list)), \
                        f"boxes_seq should be a tuple/list but got {type(parsed)}"
                    for elem in parsed:
                        assert isinstance(elem, (tuple, list)), \
                            f"the elem in boxes_seq should be a tuple/list but got {type(elem)} for elem: {elem}"
                        assert len(elem) != 0, \
                            f"the elem in boxes_seq should not be empty."
                        for atom in elem:
                            assert isinstance(atom, int), \
                                f"the boxes_seq atom should be a int idx but got {type(atom)} for atom: {atom}"
                except (AssertionError, SyntaxError) as e:
                    raise gr.Error(f"error when parse boxes_seq_str: {str(e)} for input: {boxes_seq_str}")
                return parsed

            boxes_seq = parse_boxes_seq(boxes_seq)

            mm_state = state['ibs'].to_model()
            ds = state['ds']
            print(mm_state)
            if 'image' in mm_state and bool(mm_state['image']):
                # multimodal mode
                if ds.image is not None and ds.image != mm_state['image']:
                    raise gr.Error("shikra only support single image conversation but got different images. maybe u want `Reset Dialogue`")
                if ds.image != mm_state['image']:
                    ds.set_image(mm_state['image'])

                def validate_message_box(user_input: str, boxes_seq: list, boxes_value: list):
                    if boxes_value and (not boxes_seq):
                        grWarning("has box drawn but set no boxes_seq")

                    if boxes_seq and (not boxes_value):
                        grWarning("ignored boxes_seq because no box drawn.")

                    boxes_placeholder_num = str(user_input).count('<boxes>')
                    if boxes_placeholder_num != len(boxes_seq):
                        raise gr.Error(f"<boxes> and boxes_seq num not match: {boxes_placeholder_num} {len(boxes_seq)}")

                    for boxes in boxes_seq:
                        for bidx in boxes:
                            if not (0 <= bidx < len(boxes_value)):
                                raise gr.Error(f"boxes_seq out of range: {boxes_seq} {len(boxes_value)}")

                try:
                    validate_message_box(user_input, boxes_seq, mm_state['boxes'])
                    ds.append_message(role=ds.roles[0], message=user_input, boxes=mm_state['boxes'], boxes_seq=boxes_seq)
                except Exception as e:
                    raise gr.Error(f"error when append message: {str(e)}")
            else:
                # text-only mode
                if bool(boxes_seq):
                    grWarning("ignored boxes_seq in text-only mode")
                boxes_placeholder_num = str(user_input).count('<boxes>')
                if boxes_placeholder_num:
                    gr.Error("use <boxes> in input but no image found.")
                ds.append_message(role=ds.roles[0], message=user_input)

            model_inputs = ds.to_model_input()
            model_inputs['images'] = model_inputs['images'].to(torch.float16)
            print(f"model_inputs: {model_inputs}")

            if do_sample:
                gen_kwargs = dict(
                    use_cache=True,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_length,
                    top_p=top_p,
                    temperature=float(temperature),
                )
            else:
                gen_kwargs = dict(
                    use_cache=True,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_length,
                )
            print(gen_kwargs)
            input_ids = model_inputs['input_ids']
            st_time = time.time()
            with torch.inference_mode():
                with torch.autocast(dtype=torch.float16, device_type='cuda'):
                    output_ids = model.generate(**model_inputs, **gen_kwargs)
            print(f"done generated in {time.time() - st_time} seconds")
            input_token_len = input_ids.shape[-1]
            response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            print(f"response: {response}")

            # update new message

            def build_boxes_image(text, image):
                if image is None:
                    return text, None
                print(text, image)
                import re

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
                from torchvision.transforms import ToPILImage
                res = ToPILImage()(res)
                _, path = tempfile.mkstemp(suffix='.jpg', dir=TEMP_FILE_DIR)
                res.save(path)
                add_submit_temp_image(state, path)

                # post process text color
                print(text)
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
                return text, path

            def convert_one_round_message(conv, image=None):
                text_query = f"{conv[0][0]}: {conv[0][1]}"
                text_answer = f"{conv[1][0]}: {conv[1][1]}"
                text_query, image_query = build_boxes_image(text_query, image)
                text_answer, image_answer = build_boxes_image(text_answer, image)

                new_chat = []
                new_chat.append([parse_text(text_query), None])
                if image_query is not None:
                    new_chat.append([(image_query,), None])

                new_chat.append([None, parse_text(text_answer)])
                if image_answer is not None:
                    new_chat.append([None, (image_answer,)])
                return new_chat

            ds.append_message(role=ds.roles[1], message=response)
            conv = ds.to_gradio_chatbot_new_messages()
            new_message = convert_one_round_message(conv, image=mm_state.get('image', None))
            print(new_message)
            state['_submit_new_message'] = new_message
            return state, chatbot


        def submit_step2(state, user_input, boxes_seq, chatbot):
            if '_submit_new_message' in state:
                chatbot.extend(state['_submit_new_message'])
                del state['_submit_new_message']
                return state, None, None, chatbot
            return state, user_input, boxes_seq, chatbot


        submitBtn.click(
            submit_step1,
            [state, template, user_input, boxes_seq, chatbot, do_sample, max_length, top_p, temperature],
            [state, chatbot],
        ).then(
            submit_step2,
            [state, user_input, boxes_seq, chatbot],
            [state, user_input, boxes_seq, chatbot],
        )

    print("launching...")
    demo.queue().launch(server_name=args.server_name, server_port=args.server_port)
