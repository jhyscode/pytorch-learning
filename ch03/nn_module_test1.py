# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 16:30
# @Author  : jhys
# @FileName: nn_module_test1.py

import torch.nn as nn
from collections import OrderedDict


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dense_block = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out


model = MyNet()
print(model)
