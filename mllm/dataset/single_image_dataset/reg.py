from ..utils import (
    MInstrDataset,
)

from ..root import (
    DATASETS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER,
)


@DATASETS.register_module()
class REGDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        bbox = item['bbox']

        image = self.get_image(img_path)
        question = self.get_template().replace(OBJS_PLACEHOLDER, BOXES_PLACEHOLDER)
        caption = expr

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                    'boxes_seq': [[0]],
                },
                {
                    'from': 'gpt',
                    'value': f'{caption}',
                }
            ]
        }
        return ret


@DATASETS.register_module()
class GCDataset(REGDataset):
    pass
