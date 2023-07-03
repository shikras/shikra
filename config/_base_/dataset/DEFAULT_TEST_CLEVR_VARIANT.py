CLEVR_TEST_COMMON_CFG = dict(
    type='ClevrDataset',
    filename=r'{{fileDirname}}/../../../data/CLEVR_val_questions_with_ans.jsonl',
    image_folder=r'zz1424:s3://publicdataset_33/CLEVR/unzip/CLEVR_v1.0/images/val',
    scene_graph_file=None,
)

DEFAULT_TEST_CLEVR_VARIANT = dict(
    CLEVR_A_VAL=dict(
        **CLEVR_TEST_COMMON_CFG,
        version='q-a',
        template_file=r"{{fileDirname}}/template/VQA.json",
    ),
    CLEVR_S_VAL=dict(
        **CLEVR_TEST_COMMON_CFG,
        version='q-a',
        template_file=r"{{fileDirname}}/template/VQA_CoT.json",
    ),
    CLEVR_BS_VAL=dict(
        **CLEVR_TEST_COMMON_CFG,
        version='q-a',
        template_file=r"{{fileDirname}}/template/VQA_PCoT.json",
    ),
)
