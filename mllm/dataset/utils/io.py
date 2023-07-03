import sys
import time
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def read_img_general(img_path):
    if "s3://" in img_path:
        cv_img = read_img_ceph(img_path)
        # noinspection PyUnresolvedReferences
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    else:
        return Image.open(img_path).convert('RGB')


client = None


def read_img_ceph(img_path):
    init_ceph_client_if_needed()
    img_bytes = client.get(img_path)
    assert img_bytes is not None, f"Please check image at {img_path}"
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    # noinspection PyUnresolvedReferences
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa
        client = Client(enable_mc=True)
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")