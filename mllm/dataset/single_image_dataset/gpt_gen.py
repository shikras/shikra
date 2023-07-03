from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
)
from ..utils import MInstrDataset
from ..utils.flickr30k_entities_utils import PHRASE_ST_PLACEHOLDER, PHRASE_ED_PLACEHOLDER


@DATASETS.register_module()
class GPT4Gen(MInstrDataset):
    def __init__(self, *args, version, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.version = version
        assert version in ['a', 'c', 'bc']

    def __getitem__(self, item):
        raw = self.get_raw_item(item)
        #
        image = self.get_image(raw['img_path'])
        #
        boxes = raw['boxes']
        #
        question = raw['question']
        question = question.replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)
        query_boxes_seq = raw['question_boxes_seq']

        if self.version == 'a':
            final_answer = raw['answer']
            answer_boxes_seq = None
        elif self.version == 'c':
            final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, '')
            answer_boxes_seq = None
        elif self.version == 'bc':
            final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, '').replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER)
            answer_boxes_seq = raw['answer_boxes_seq']
        else:
            assert False

        ret = {
            'image': image,
            'target': {'boxes': boxes},
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                    'boxes_seq': query_boxes_seq,
                },
                {
                    'from': 'gpt',
                    'value': final_answer,
                    'boxes_seq': answer_boxes_seq,
                }
            ]
        }
        return ret
