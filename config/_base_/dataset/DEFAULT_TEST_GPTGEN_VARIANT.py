GPTGEN_TEST_COMMON_CFG = dict(
    type='GPT4Gen',
    filename=r'{{fileDirname}}/../../../data/GPT4GEN_BoxCoT_test.jsonl',
    image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
)

DEFAULT_TEST_GPTGEN_VARIANT = dict(
    GPT4GEN_QA=dict(**GPTGEN_TEST_COMMON_CFG, version='a', template_file=r"{{fileDirname}}/template/VQA.json"),
    GPT4GEN_QC=dict(**GPTGEN_TEST_COMMON_CFG, version='c', template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GPT4GEN_QBC=dict(**GPTGEN_TEST_COMMON_CFG, version='bc', template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),
)
