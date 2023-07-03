FLICKR_TEST_COMMON_CFG = dict(
    type='FlickrDataset',
    image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
    max_dynamic_size=None,
)

DEFAULT_TEST_FLICKR_VARIANT = dict(
    FLICKR_EVAL_with_box=dict(
        **FLICKR_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_eval.jsonl',
        template_file=r'{{fileDirname}}/template/flickr30k.json',
    ),
    FLICKR_EVAL_without_box=dict(
        **FLICKR_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_eval.jsonl',
        template_file=r'{{fileDirname}}/template/image_cap.json',
    ),
    FLICKR_TEST_with_box=dict(
        **FLICKR_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_test.jsonl',
        template_file=r'{{fileDirname}}/template/flickr30k.json',
    ),
    FLICKR_TEST_without_box=dict(
        **FLICKR_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_test.jsonl',
        template_file=r'{{fileDirname}}/template/image_cap.json',
    ),
)
