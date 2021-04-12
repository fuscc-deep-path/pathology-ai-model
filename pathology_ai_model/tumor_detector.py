# -*- coding: utf-8 -*-
"""
    pathology_ai_model.tumor_detector
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Detect tumor type from the image patch.

    :copyright: © 2019 by the Choppy Team.
    :license: AGPLv3+, see LICENSE for more details.
"""

import os
import csv
import torch
import numpy as np
import torch.nn.functional as F

from glob import glob
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import models, transforms
from pathology_ai_model.utils import get_modelpath


class TumorDetPatchReader(data.Dataset):
    def __init__(self, folder, format):
        self.lists = glob(os.path.join(folder, '*' + format))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.8201, 0.5207, 0.7189], [0.1526, 0.1542, 0.1183])])

    def __getitem__(self, index):
        name = self.lists[index]
        img = Image.open(name)  # .convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, -1, name.split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.lists)


def load_net(modelpath, numclasses=2):
    net = models.resnet18(pretrained=False, num_classes=numclasses)
    net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath).items()})
    net = net.cuda()   # Only one cuda can be use
    return net.eval()


# hook the feature extractor
features_blobs = [0]
def hook_feature(module, input, output):
    # features_blobs.append(output.data.cpu().numpy())
    features_blobs[0] = output.data.cpu().numpy()


def net_pred_extract_feats(loader, net, cls=2):
    """Network and image for prediction and extract each patch's feats,
       the npz file of tumor detcetion results and the several .csv files are in the same folder
    """
    net._modules.get('avgpool').register_forward_hook(hook_feature)  # get feature maps

    score = np.empty([0, cls])
    bin = np.array([])
    namelist = np.array([])

    with torch.no_grad():
        for i, (img, _, name) in tqdm(enumerate(loader)):
            img = img.cuda()
            predProb = F.softmax(net(img), dim=1)
            predBina = torch.argmax(predProb, dim=1)

            bin = np.concatenate((bin, predBina.cpu().numpy()), axis=0)
            score = np.concatenate((score, predProb.cpu().numpy()), axis=0)
            namelist = np.concatenate((namelist, name), axis=0)

    return score, bin, namelist


def start_model(datapath, feats_savepath):
    modelpath = get_modelpath('norm_model_epoch_105.pkl')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('Loading model...')
    net = load_net(modelpath, numclasses=5)

    dataset = TumorDetPatchReader(datapath, format='png')
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

    scores, bins, namelist = net_pred_extract_feats(loader, net, cls=5)
    print(len(scores), '\t', len(bins), '\t', (namelist))

    np.savez(feats_savepath, score=scores, bin=bins, namelist=namelist)
