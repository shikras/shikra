# server 端---------------------------------------
import argparse
import os
import sys
import base64
import logging
import time
from pathlib import Path
from io import BytesIO

import torch
import uvicorn
import transformers
from PIL import Image
from mmengine import Config
from transformers import BitsAndBytesConfig
from fastapi import FastAPI, Request, HTTPException

sys.path.append(str(Path(__file__).parent.parent.parent))

from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.models.builder.build_shikra import load_pretrained_shikra
from mllm.dataset.utils.transform import expand2square, box_xyxy_expand2square

log_level = logging.DEBUG
transformers.logging.set_verbosity(log_level)
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

#########################################
# mllm model init
#########################################
parser = argparse.ArgumentParser("Shikra Server Demo")
parser.add_argument('--model_path', required=True)
parser.add_argument('--load_in_8bit', action='store_true')
parser.add_argument('--server_name', default='127.0.0.1')
parser.add_argument('--server_port', type=int, default=12345)

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
print(
    f"LLM device: {model.device}, is_quantized: {getattr(model, 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
print(
    f"vision device: {model.model.vision_tower[0].device}, is_quantized: {getattr(model.model.vision_tower[0], 'is_quantized', False)}, is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}, is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")

preprocessor['target'] = {'boxes': PlainBoxFormatter()}
tokenizer = preprocessor['text']

#########################################
# fast api
#########################################
app = FastAPI()


@app.post("/shikra")
async def shikra(request: Request):
    try:
        # receive parameters
        para = await request.json()
        img_base64 = para["img_base64"]
        user_input = para["text"]
        boxes_value = para.get('boxes_value', [])
        boxes_seq = para.get('boxes_seq', [])

        do_sample = para.get('do_sample', False)
        max_length = para.get('max_length', 512)
        top_p = para.get('top_p', 1.0)
        temperature = para.get('temperature', 1.0)

        # parameters preprocess
        pil_image = Image.open(BytesIO(base64.b64decode(img_base64))).convert("RGB")
        ds = prepare_interactive(model_args, preprocessor)

        image = expand2square(pil_image)
        boxes_value = [box_xyxy_expand2square(box, w=pil_image.width, h=pil_image.height) for box in boxes_value]

        ds.set_image(image)
        ds.append_message(role=ds.roles[0], message=user_input, boxes=boxes_value, boxes_seq=boxes_seq)
        model_inputs = ds.to_model_input()
        model_inputs['images'] = model_inputs['images'].to(torch.float16)
        print(f"model_inputs: {model_inputs}")

        # generate
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
        response = tokenizer.batch_decode(output_ids[:, input_token_len:])[0]
        print(f"response: {response}")

        input_text = tokenizer.batch_decode(input_ids)[0]
        return {
            "input": input_text,
            "response": response,
        }

    except Exception as e:
        logging.exception(str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=args.server_name, port=args.server_port, log_level="info")
