_base_ = ['_base_/dataset/DEFAULT_TEST_DATASET.py', '_base_/model/shikra.py', '_base_/train/eval.py']

training_args = dict(
    output_dir='./exp/{{fileBasenameNoExtension}}',

    do_train=False,
    do_eval=False,
    do_predict=False,
    do_multi_predict=True,

    fp16=True,
    fp16_full_eval=True,
    bf16=False,
    bf16_full_eval=False,
    per_device_eval_batch_size=8,
)

model_args = dict(
    model_name_or_path=None,
)

data_args = dict(
    train=None,
    validation=None,
    test=None,
    multitest={k: {'cfg': v, 'compute_metric': dict(type='GQAComputeMetrics')} for k, v in _base_.DEFAULT_TEST_POPE_VARIANT.items() if 'q_a' in k},

    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
