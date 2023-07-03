from ..root import DATASETS, IMAGE_PLACEHOLDER
from ..utils import MInstrDataset


@DATASETS.register_module()
class CaptionDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        caption = item['caption']

        image = self.get_image(img_path)
        question = self.get_template()

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': caption,
                }
            ]
        }
        return ret
