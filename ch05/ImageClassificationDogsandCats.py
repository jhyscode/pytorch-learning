# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 20:41
# @Author  : jhys
# @FileName: ImageClassificationDogsandCats.py

from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import time


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)


is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

simple_transform = transforms.Compose([transforms.Resize((224, 224))
                                          , transforms.ToTensor()
                                          , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
train = ImageFolder('dogsandcats/train/', simple_transform)
valid = ImageFolder('dogsandcats/valid/', simple_transform)

print(train.class_to_idx)
print(train.classes)
