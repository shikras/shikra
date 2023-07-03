_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    #
    train=dict(
        type='ConcatDataset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.rec}},
            {{_base_.DEFAULT_TRAIN_DATASET.recvg}},
        ],
    ),
    validation=None,
    test=None,

    # compute_metric
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
