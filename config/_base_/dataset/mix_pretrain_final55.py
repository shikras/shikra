_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    #
    train=dict(
        type='InterleaveDateset',
        probabilities=[0.5, 0.5],
        seed=None,
        stopping_strategy='first_exhausted',
        cfgs=[
            dict(
                type='ConcatDatasetWithShuffle',
                cfgs=[
                    {{_base_.DEFAULT_TRAIN_DATASET.instruct}},
                    {{_base_.DEFAULT_TRAIN_DATASET.GPT4GEN_QBC}},
                    {{_base_.DEFAULT_TRAIN_DATASET.GPT4GEN_RD_QBC}},
                ]
            ),
            dict(
                type='InterleaveDateset',
                probabilities=[1 / 7] * 7,
                seed=None,
                stopping_strategy='first_exhausted',
                cfgs=[
                    {{_base_.DEFAULT_TRAIN_DATASET.flickr}},
                    {{_base_.DEFAULT_TRAIN_DATASET.rec}},
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=43,
                        cfgs=[
                            {{_base_.DEFAULT_TRAIN_DATASET.VQAv2_train}},
                            {{_base_.DEFAULT_TRAIN_DATASET.VQAE_train}},
                            {{_base_.DEFAULT_TRAIN_DATASET.VQAX_train}},
                        ],
                    ),
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=44,
                        cfgs=[
                            {{_base_.DEFAULT_TRAIN_DATASET.VCR_q_ra}},
                            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qc_rac}},
                            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qac_r}},
                        ],
                    ),
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=45,
                        portion=3,
                        cfgs=[
                            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_b}},
                            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_p}},
                        ]
                    ),
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=46,
                        cfgs=[
                            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_oq_bp}},
                            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_sq_bp}},
                            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_gq_bp}},
                            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_p}},
                            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_b}},
                        ]
                    ),
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=47,
                        cfgs=[
                            {{_base_.DEFAULT_TRAIN_DATASET.reg}},
                            {{_base_.DEFAULT_TRAIN_DATASET.caption}},
                            dict(
                                type='SubSet',
                                portion=2 / 3,
                                do_shuffle=True,
                                seed=40,
                                cfg={{_base_.DEFAULT_TRAIN_DATASET.llavacc3m}},
                            ),
                            dict(
                                type='SubSet',
                                portion=2 / 3,
                                do_shuffle=True,
                                seed=41,
                                cfg={{_base_.DEFAULT_TRAIN_DATASET.llavalcs}},
                            ),
                            dict(
                                type='SubSet',
                                portion=1 / 15,
                                do_shuffle=True,
                                seed=42,
                                cfg={{_base_.DEFAULT_TRAIN_DATASET.gc}},
                            ),
                        ]
                    )
                ],
            )
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
