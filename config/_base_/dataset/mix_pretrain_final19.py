_base_ = ['mix_pretrain_final55.py']

data_args = dict(
    train=dict(
        probabilities=[0.1, 0.9],
    )
)
