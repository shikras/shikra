<p align="center">
  <a href="#">
<img src="./assets/logo.png" alt="Logo" width="130"></a>
  <h4 align="center"><font color="#966661">Shikra</font>: Unleashing Multimodal LLM’s Referential Dialogue Magic</h4>
  <p align="center">
    <a href='https://github.com/shikras/shikra'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='http://arxiv.org/abs/2306.15195'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
  </p>
</p>


***
<font color="#966661">**Shikra**</font>, an MLLM designed to kick off **referential dialogue** by excelling in spatial coordinate inputs/outputs in natural language, **without** additional vocabularies, position encoders, pre-/post-detection, or external plug-in models.

<p align="center"><img src="./assets/teaser.jpg" alt="teaser" width="300px" /></p>

## News
[07/03] We released the code, [data](https://drive.google.com/file/d/1CNLu1zJKPtliQEYCZlZ8ykH00ppInnyN/view?usp=drive_link) and [Shikra-7B checkpoint](https://huggingface.co/shikras/shikra-7b-delta-v1).

[06/28] We released **Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic**, which is designed to kick off **referential dialogue**. Checkout the [paper](https://arxiv.org/abs/2306.15195).

## Contents

- [Install](#install)
- [Shikra weights](#shikra-weights)
- [Dataset](https://github.com/shikras/shikra/blob/main/docs/data.md)

## Install

```shell
conda create -n shikra python=3.10
conda activate shikra
pip install -r requirements.txt
```

### configure accelerate

```shell
accelerate config
```

## Shikra weights

We release Shikra weights as delta weights to comply with the LLaMA model license. You can add our delta to the original LLaMA weights to obtain the Shikra weights.

Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get weights by applying our delta ([shikra-7b-delta](https://huggingface.co/shikras/shikra-7b-delta-v1)). It will automatically download delta weights from our Hugging Face account.

```shell
python mllm/models/shikra/apply_delta.py \
    --base /path/to/llama-7b \
    --target /output/path/to/shikra-7b \
    --delta shikras/shikra-7b-delta-v1
```

## Train

After preparing [data](https://github.com/shikras/shikra/blob/main/docs/data.md), you can train the model using the command:

```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_pretrain_final19_stage2.py \
        --cfg-options model_args.model_name_or_path=/path/to/init/checkpoint
```

mmengine style args and huggingface:Trainer args are supported. For example, you can change epoch and output_dir like this:

```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_pretrain_final19_stage2.py \
        --cfg-options model_args.model_name_or_path=/path/to/init/checkpoint \
        --num_train_epochs 3 \
        --output_dir /path/to/new/exp/dir
```

where `--cfg-options a=balabala b=balabala` is mmengine style argument. They will overwrite the argument predefined in config file. And `--num_train_epochs` , `--output_dir` are huggingface:Trainer argument.

## Inference

After preparing [data](https://github.com/shikras/shikra/blob/main/docs/data.md), you can inference the model using the command:

```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_eval_multi_pope.py \
        --cfg-options model_args.model_name_or_path=/path/to/checkpoint
```

mmengine style args and huggingface:Trainer args are supported. for example, you can change eval batchsize like this:

```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_eval_multi_pope.py \
        --cfg-options model_args.model_name_or_path=/path/to/checkpoint \
        --per_device_eval_batch_size 1
```

where `--cfg-options a=balabala b=balabala` is mmengine style argument. They will overwrite the argument predefined in config file. And `--per_device_eval_batch_size` is huggingface:Trainer argument.

the prediction result will be saved in `output_dir/multitest_xxxx_extra_prediction.jsonl`, which hold the same order as the input dataset. 

## Examples

<img src="./assets/shikra_case_1.jpg" alt="shikra_case_1" style="zoom: 25%;" />

## Cite

```bibtex
@article{chen2023shikra,
  title={Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic},
  author={Chen, Keqin and Zhang, Zhao and Zeng, Weili and Zhang, Richong and Zhu, Feng and Zhao, Rui},
  journal={arXiv preprint arXiv:2306.15195},
  year={2023}
}
```

## Acknowledgement

This repo benefits from [LLaVA](https://github.com/haotian-liu/LLaVA), [Vicuna](https://github.com/lm-sys/FastChat) and [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning). Thanks for their wonderful works.
