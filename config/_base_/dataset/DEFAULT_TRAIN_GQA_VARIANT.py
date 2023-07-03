GQA_TRAIN_COMMON_CFG = dict(
    type='GQADataset',
    filename=r'{{fileDirname}}/../../../data/gqa_question_balanced_with_cot.jsonl',
    image_folder=r'zz1424:s3://publicdataset_11/GQA/unzip/images',
    scene_graph_file=r"{{fileDirname}}/../../../data/gqa_scene_graph_data.jsonl",
    scene_graph_index=r"{{fileDirname}}/../../../data/gqa_scene_graph_index.json",
)

DEFAULT_TRAIN_GQA_VARIANT = dict(
    GQA_Q_A=dict(**GQA_TRAIN_COMMON_CFG, version="q-a", template_file=r"{{fileDirname}}/template/VQA.json"),
    GQA_Q_C=dict(**GQA_TRAIN_COMMON_CFG, version="q-c", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_Q_BC=dict(**GQA_TRAIN_COMMON_CFG, version="q-bc", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    GQA_Q_S=dict(**GQA_TRAIN_COMMON_CFG, version="q-s", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_Q_BS=dict(**GQA_TRAIN_COMMON_CFG, version="q-bs", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    GQA_Q_L=dict(**GQA_TRAIN_COMMON_CFG, version="q-l", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_Q_BL=dict(**GQA_TRAIN_COMMON_CFG, version="q-bl", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),

    GQA_QB_A=dict(**GQA_TRAIN_COMMON_CFG, version="qb-a", template_file=r"{{fileDirname}}/template/VQA.json"),
    GQA_QB_C=dict(**GQA_TRAIN_COMMON_CFG, version="qb-c", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QB_BC=dict(**GQA_TRAIN_COMMON_CFG, version="qb-bc", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    GQA_QB_S=dict(**GQA_TRAIN_COMMON_CFG, version="qb-s", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QB_BS=dict(**GQA_TRAIN_COMMON_CFG, version="qb-bs", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    GQA_QB_L=dict(**GQA_TRAIN_COMMON_CFG, version="qb-l", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QB_BL=dict(**GQA_TRAIN_COMMON_CFG, version="qb-bl", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),

    GQA_QBP_A=dict(**GQA_TRAIN_COMMON_CFG, version="qbp-a", template_file=r"{{fileDirname}}/template/VQA.json"),
    GQA_QBP_C=dict(**GQA_TRAIN_COMMON_CFG, version="qbp-c", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QBP_BC=dict(**GQA_TRAIN_COMMON_CFG, version="qbp-bc", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    GQA_QBP_S=dict(**GQA_TRAIN_COMMON_CFG, version="qbp-s", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QBP_BS=dict(**GQA_TRAIN_COMMON_CFG, version="qbp-bs", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
    GQA_QBP_L=dict(**GQA_TRAIN_COMMON_CFG, version="qbp-l", template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GQA_QBP_BL=dict(**GQA_TRAIN_COMMON_CFG, version="qbp-bl", template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
)
