POINT_TEST_COMMON_CFG_LOCAL = dict(
    type='Point_QA_local',
    image_folder='zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data',
    template_file=r"{{fileDirname}}/template/VQA.json",
)

POINT_TEST_COMMON_CFG_TWICE = dict(
    type='Point_QA_twice',
    image_folder='zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data',
    template_file=r"{{fileDirname}}/template/VQA.json",
)

POINT_TEST_COMMON_CFG_V7W = dict(
    type='V7W_POINT',
    image_folder='sh41:s3://MultiModal/Monolith/academic/v7w/data',
    template_file=r"{{fileDirname}}/template/VQA.json",
    do_shuffle_choice=True,
)

DEFAULT_TEST_POINT_VARIANT = dict(
    POINT_LOCAL_b_val=dict(**POINT_TEST_COMMON_CFG_LOCAL, version='b', filename='{{fileDirname}}/../../../data/pointQA_local_val.jsonl'),
    POINT_LOCAL_p_val=dict(**POINT_TEST_COMMON_CFG_LOCAL, version='p', filename='{{fileDirname}}/../../../data/pointQA_local_val.jsonl'),
    POINT_TWICE_oq_b_val=dict(**POINT_TEST_COMMON_CFG_TWICE, version='oq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    POINT_TWICE_oq_p_val=dict(**POINT_TEST_COMMON_CFG_TWICE, version='oq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    POINT_TWICE_sq_b_val=dict(**POINT_TEST_COMMON_CFG_TWICE, version='sq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    POINT_TWICE_sq_p_val=dict(**POINT_TEST_COMMON_CFG_TWICE, version='sq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    POINT_TWICE_gq_b_val=dict(**POINT_TEST_COMMON_CFG_TWICE, version='gq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    POINT_TWICE_gq_p_val=dict(**POINT_TEST_COMMON_CFG_TWICE, version='gq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    POINT_V7W_p_val=dict(**POINT_TEST_COMMON_CFG_V7W, version='p', filename='{{fileDirname}}/../../../data/v7w_pointing_val.jsonl'),
    POINT_V7W_b_val=dict(**POINT_TEST_COMMON_CFG_V7W, version='b', filename='{{fileDirname}}/../../../data/v7w_pointing_val.jsonl'),

    POINT_LOCAL_b_test=dict(**POINT_TEST_COMMON_CFG_LOCAL, version='b', filename='{{fileDirname}}/../../../data/pointQA_local_test.jsonl'),
    POINT_LOCAL_p_test=dict(**POINT_TEST_COMMON_CFG_LOCAL, version='p', filename='{{fileDirname}}/../../../data/pointQA_local_test.jsonl'),
    POINT_TWICE_oq_b_test=dict(**POINT_TEST_COMMON_CFG_TWICE, version='oq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    POINT_TWICE_oq_p_test=dict(**POINT_TEST_COMMON_CFG_TWICE, version='oq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    POINT_TWICE_sq_b_test=dict(**POINT_TEST_COMMON_CFG_TWICE, version='sq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    POINT_TWICE_sq_p_test=dict(**POINT_TEST_COMMON_CFG_TWICE, version='sq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    POINT_TWICE_gq_b_test=dict(**POINT_TEST_COMMON_CFG_TWICE, version='gq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    POINT_TWICE_gq_p_test=dict(**POINT_TEST_COMMON_CFG_TWICE, version='gq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    POINT_V7W_p_test=dict(**POINT_TEST_COMMON_CFG_V7W, version='p', filename='{{fileDirname}}/../../../data/v7w_pointing_test.jsonl'),
    POINT_V7W_b_test=dict(**POINT_TEST_COMMON_CFG_V7W, version='b', filename='{{fileDirname}}/../../../data/v7w_pointing_test.jsonl'),
)
