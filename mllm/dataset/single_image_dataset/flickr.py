from torch.utils.data import Dataset

from ..root import DATASETS, BOXES_PLACEHOLDER, IMAGE_PLACEHOLDER
from ..utils import MInstrDataset
from ..utils.flickr30k_entities_utils import (
    flatten_annotation,
    PHRASE_ED_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER,
)


class FlickrParser(Dataset):
    def __init__(self, filename, annotation_dir):
        self.filename = filename
        self.annotation_dir = annotation_dir

        self.indexes = [line.strip() for line in open(filename, 'r', encoding='utf8')]
        self.data = flatten_annotation(self.annotation_dir, self.indexes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def dump(self, filename):
        import json
        with open(filename, 'w', encoding='utf8') as f:
            for obj in self.data:
                obj_str = json.dumps(obj)
                f.write(obj_str)
                f.write('\n')


@DATASETS.register_module()
class FlickrDataset(MInstrDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = f"{item['image_id']}.jpg"
        caption = item['sentence']

        image = self.get_image(img_path)
        caption = caption.replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
        question = self.get_template()

        ret = {
            'image': image,
            'target': {'boxes': item['boxes']},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': caption,
                    'boxes_seq': item['boxes_seq'],
                }
            ]
        }
        return ret
