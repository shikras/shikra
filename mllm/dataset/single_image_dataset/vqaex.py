from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
)
from ..utils import MInstrDataset


@DATASETS.register_module()
class VQAEXDataset(MInstrDataset):
    def __init__(self, *args, is_e_dataset: bool, has_annotation=True, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.has_annotation = has_annotation
        self.is_e_dataset = is_e_dataset

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        image = self.get_image(image_path=item['file_path'])

        question = item['question']
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)

        if self.has_annotation:
            if self.is_e_dataset:
                final_answer = ""
                final_answer += item['explanation'][0]
                final_answer += f" So the answer is {item['multiple_answers']}."
            else:
                final_answer = ""
                final_answer += "".join(item['justification'])
                final_answer += f" So the answer is {item['multiple_choice_answer']}."
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
                    'value': final_answer,
                },
            ]
        }
        return ret
