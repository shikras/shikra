model_args = dict(
    type='shikra',
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=None,
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
        tokenize_kwargs=dict(truncation_size=2048),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)