from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
)
from ..utils import MInstrDataset


@DATASETS.register_module()
class POPEVQADataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        image = self.get_image(image_path=item['image'])

        question = item['text']
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        label = str(item['label']).lower()

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': f"The answer is {label} .",
                },
            ]
        }
        return ret
