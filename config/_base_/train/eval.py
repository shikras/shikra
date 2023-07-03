training_args = dict(
    # run
    output_dir=None,  # required. must be filled by derived configs.
    overwrite_output_dir=False,
    report_to='none',
    seed=42,

    # datasets
    remove_unused_columns=False,

    # logging
    logging_steps=1,

    # eval and predict
    do_eval=True,
    do_predict=True,
    bf16=True,
    bf16_full_eval=True,
    predict_with_generate=True,
    per_device_eval_batch_size=16,
    # dataloader_num_workers=4,
)
