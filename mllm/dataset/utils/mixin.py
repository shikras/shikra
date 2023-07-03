import json
import os

import numpy as np
from torch.utils.data import Dataset

from .io import read_img_general


class QuestionTemplateMixin:
    def __init__(
            self,
            *args,
            template_string=None,
            template_file=None,
            max_dynamic_size=None,
            placeholders=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.template_string = template_string
        self.template_file = template_file
        self.max_dynamic_size = max_dynamic_size
        self.placeholders = placeholders
        if template_string is None and template_file is None:
            raise ValueError("assign either template_string or template_file")
        if template_string is not None and template_file is not None:
            raise ValueError(f"assign both template_string and template_file:\nstring:{template_string}\nfile:{template_file}")
        if template_string is not None:
            self.templates = [self.template_string]
        else:
            assert template_file is not None
            self.templates = json.load(open(template_file, 'r', encoding='utf8'))
        if self.max_dynamic_size is not None:
            self.templates = self.templates[: self.max_dynamic_size]

        # sanity check
        assert self.placeholders is not None
        for template in self.templates:
            for placeholder in placeholders:
                assert str(template).count(placeholder) == 1, f"template: {template}\nplaceholder:{placeholder}"

    def get_template(self):
        import random
        return random.choice(self.templates)

    def template_nums(self):
        return len(self.templates)


class MInstrDataset(QuestionTemplateMixin, Dataset):
    _repr_indent = 4

    def __init__(self, filename, image_folder=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.image_folder = image_folder
        self.rng = np.random.default_rng(seed)

        self.data = []
        with open(filename, 'r', encoding='utf8') as f:
            # for line in tqdm(f, desc=f'{self.__class__.__name__} loading ann {self.filename}'):
            for line in f:
                self.data.append(line)

    def get_raw_item(self, index):
        return json.loads(self.data[index])

    def get_image(self, image_path):
        if self.image_folder is not None:
            image_path = os.path.join(self.image_folder, image_path)
        image = read_img_general(image_path)
        return image

    def get_template(self):
        return self.rng.choice(self.templates)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
            f"ann file: {self.filename}"
        ]
        if self.image_folder is not None:
            body.append(f"image folder: {self.image_folder}")
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    # noinspection PyMethodMayBeStatic
    def extra_repr(self) -> str:
        return ""
