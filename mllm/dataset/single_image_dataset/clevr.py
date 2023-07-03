import json

from ..root import DATASETS, IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER, POINTS_PLACEHOLDER
from ..utils import MInstrDataset


@DATASETS.register_module()
class ClevrDataset(MInstrDataset):
    def __init__(self, *args, scene_graph_file, version, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.scene_graph_file = scene_graph_file
        self.version = version
        qtype, atype = version.split('-')
        assert qtype in ['q']
        assert atype in ['a', 's', 'bs']
        self.qtype = qtype
        self.atype = atype

        if scene_graph_file is None:
            self.scene_graph = None
        else:
            self.scene_graph = [line for line in open(scene_graph_file, 'r', encoding='utf8')]

    def get_raw_item(self, index):
        question = json.loads(self.data[index])
        if self.scene_graph is None:
            scene = None
        else:
            scene = json.loads(self.scene_graph[question['image_index']])
        return question, scene

    def __getitem__(self, index):
        question, scene = self.get_raw_item(index)
        img_path = question['image_filename']
        image = self.get_image(img_path)

        if self.atype == 'a':
            boxes = []
            answer = f"The answer is {question['answer']}."
            answer_boxes_seq = []
        elif self.atype == 's':
            answer, boxes, answer_boxes_seq = clevr_ss_cot(obj=question, scene=scene, add_ref=False)
            answer += f" The answer is {question['answer']}."
        elif self.atype == 'bs':
            answer, boxes, answer_boxes_seq = clevr_ss_cot(obj=question, scene=scene, add_ref=True)
            answer += f" The answer is {question['answer']}."
        else:
            assert False

        if self.qtype == 'q':
            query_boxes_seq = []
            final_query = self.get_template().replace(QUESTION_PLACEHOLDER, question['question'])
        else:
            assert False

        ret = {
            'image': image,
            'target': {'points': boxes},
            'conversations': [
                {
                    'from': 'human',
                    'value': final_query,
                    'points_seq': query_boxes_seq,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                    'points_seq': answer_boxes_seq,
                }
            ]
        }
        return ret


def get_boxes_idx(boxes_list, refs):
    def get_idx(boxes_list, box):
        if box in boxes_list:
            return boxes_list.index(box)
        else:
            boxes_list.append(box)
            return len(boxes_list) - 1

    idx = [get_idx(boxes_list, box) for box in refs]
    return idx


def clevr_ss_cot(obj, scene, add_ref=False):
    cot = []
    boxes = []
    seq = []

    def can_add_ref():
        if p['function'] in ['unique', 'union', 'intersect', 'relate', 'same_size', 'same_shape', 'same_material', 'same_color']:
            return True
        if p['function'] in ['scene', 'filter_color', 'filter_material', 'filter_shape', 'filter_size']:
            if idx + 1 < len(obj['program']) and obj['program'][idx + 1]['function'] in ['exist', 'count']:
                return True
        return False

    for idx, p in enumerate(obj['program']):
        func = f"{p['function']}:{p['value_inputs'][0]}" if 'value_inputs' in p and p['value_inputs'] else p['function']
        inputs = f"[{','.join(map(str, p['inputs']))}]" if p['inputs'] else ""

        if add_ref and can_add_ref():
            if p['ans']:
                objs = POINTS_PLACEHOLDER
                idx = get_boxes_idx(boxes_list=boxes, refs=[scene['objects'][_]['pixel_coords'][:2] for _ in p['ans']])
                seq.append(idx)
            else:
                objs = f" Found no object."
        else:
            objs = ""
        cot.append(f"{func}{inputs}{objs}")

    ret = " -> ".join(cot)
    return ret, boxes, seq
