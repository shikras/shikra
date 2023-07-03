CLEVR_TRAIN_COMMON_CFG = dict(
    type='ClevrDataset',
    filename=r'{{fileDirname}}/../../../data/CLEVR_train_questions_with_ans.jsonl',
    image_folder=r'zz1424:s3://publicdataset_33/CLEVR/unzip/CLEVR_v1.0/images/train',
    scene_graph_file=r"{{fileDirname}}/../../../data/CLEVR_train_scenes.jsonl",
)

DEFAULT_TRAIN_CLEVR_VARIANT = dict(
    CLEVR_A=dict(
        **CLEVR_TRAIN_COMMON_CFG,
        version='q-a',
        template_file=r"{{fileDirname}}/template/VQA.json",
    ),
    CLEVR_S=dict(
        **CLEVR_TRAIN_COMMON_CFG,
        version='q-s',
        template_file=r"{{fileDirname}}/template/VQA_CoT.json",
    ),
    CLEVR_BS=dict(
        **CLEVR_TRAIN_COMMON_CFG,
        version='q-bs',
        template_file=r"{{fileDirname}}/template/VQA_PCoT.json",
    ),
)
