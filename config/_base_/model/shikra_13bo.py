_base_ = ['shikra.py']

model_args = dict(
    model_name_or_path=r'{{fileDirname}}/../../../ckpt/vicuna-13b',
    pretrain_mm_mlp_adapter=r'{{fileDirname}}/../../../ckpt/mm_projector_llava-13bv1.1-sft.bin',
)