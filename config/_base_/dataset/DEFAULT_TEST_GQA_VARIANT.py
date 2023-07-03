GQA_TEST_COMMON_CFG = dict(
    type='GQADataset',
    image_folder=r'zz1424:s3://publicdataset_11/GQA/unzip/images',
    scene_graph_file=None,
    scene_graph_index=None,
)

# use standard q-a mode
DEFAULT_TEST_GQA_VARIANT = dict(
    GQA_Q_A_BALANCED=dict(
        **GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA.json",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_balanced_questions.jsonl'
    ),
    GQA_Q_C_BALANCED=dict(
        **GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_CoT.json",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_balanced_questions.jsonl'
    ),
    GQA_Q_BC_BALANCED=dict(
        **GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_BCoT.json",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_balanced_questions.jsonl'
    ),

    GQA_Q_A=dict(
        **GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA.json",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_all_questions.jsonl',
    ),
    GQA_Q_C=dict(
        **GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_CoT.json",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_all_questions.jsonl',
    ),
    GQA_Q_BC=dict(
        **GQA_TEST_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA_BCoT.json",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_all_questions.jsonl',
    ),
)
