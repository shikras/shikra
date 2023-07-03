VQAv2_TRAIN_COMMON_CFG = dict(
    type='VQAv2Dataset',
    filename=r'{{fileDirname}}/../../../data/v2_OpenEnded_mscoco_train2014_questions.jsonl',
    image_folder=r'zz1424:s3://publicdataset_49/VQAv2/unzip/',
    template_file=r"{{fileDirname}}/template/VQA.json",
)

DEFAULT_TRAIN_VQAv2_VARIANT = dict(
    VQAv2_train=dict(**VQAv2_TRAIN_COMMON_CFG),
)