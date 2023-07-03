from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
)
from ..utils import MInstrDataset


@DATASETS.register_module()
class VQAv2Dataset(MInstrDataset):
    def __init__(self, *args, has_annotation=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.has_annotation = has_annotation

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        image = self.get_image(image_path=item['image_path'])

        question = item['question']
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        if self.has_annotation:
            final_answer = item['annotation']['multiple_choice_answer']
        else:
            final_answer = 'UNKNOWN'

        ret = {
            'image': image,
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                },
                {
                    'from': 'gpt',
                    'value': f"The answer is {final_answer}.",
                },
            ]
        }
        return ret
