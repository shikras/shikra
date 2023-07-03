_base_ = ['_base_/dataset/mix_pretrain_final55.py', '_base_/model/shikra.py', '_base_/train/shikra_fsdp.py']

training_args = dict(
    num_train_epochs=3,
    output_dir='./exp/{{fileBasenameNoExtension}}',
)

model_args = dict(
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path=None,
)
