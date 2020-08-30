# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 19:23
# @Author  : jhys
# @FileName: nn_module_test2.py

import torch
import torch.nn as nn
from collections import OrderedDict


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block = torch.nn.Sequential()
        self.conv_block.add_module("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv_block.add_module("relu1", torch.nn.ReLU())
        self.conv_block.add_module("pool1", torch.nn.MaxPool2d(2))

        self.dense_block = torch.nn.Sequential()
        self.dense_block.add_module("dense1", torch.nn.Linear(32 * 3 * 3, 128))
        self.dense_block.add_module("relu2", torch.nn.ReLU())
        self.dense_block.add_module("dense2", torch.nn.Linear(128, 10))

    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out


model = MyNet()

for i in model.children():
    print(i)
    print(type(i))
