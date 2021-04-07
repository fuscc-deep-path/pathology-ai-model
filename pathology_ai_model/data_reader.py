# -*- coding: utf-8 -*-
"""
    pathology_ai_model.data_reader
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Load data.

    :copyright: © 2019 by the Choppy Team.
    :license: AGPLv3+, see LICENSE for more details.
"""
import os
import torch
import torch.utils.data as data

from PIL import Image

class ClsDataset(data.Dataset):
    def __init__(self, txt_path, root, transform=None):
        with open(txt_path, 'r') as fh:
            lists = []
            for line in fh:
                words = line.rstrip()
                lists.append(words)

            self.lists = lists
            self.transform = transform
            self.root = root

    def __getitem__(self, index):
        imagename = self.lists[index]
        imagename = os.path.join(self.root, imagename)
        img = Image.open(imagename)

        if self.transform is not None:
            img = self.transform(img)
        return img, imagename.split('/')[-1]

    def __len__(self):
        return len(self.lists)


def deTransform(mean, std, tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor
