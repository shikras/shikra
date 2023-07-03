from ..root import DATASETS
from ..utils import MInstrDataset


@DATASETS.register_module()
class InstructDataset(MInstrDataset):
    def __init__(self, *args, add_coco_prefix=False, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(), template_string='', template_file=None)
        self.add_coco_prefix = add_coco_prefix

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        if self.add_coco_prefix:
            img_path = f"COCO_train2014_{item['image']}"
        else:
            img_path = item['image']
        conversations = item['conversations']

        image = self.get_image(img_path)
        ret = {
            'image': image,
            'conversations': conversations,
        }
        return ret
