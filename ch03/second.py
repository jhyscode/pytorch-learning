# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 15:33
# @Author  : jhys
# @FileName: second.py

import torch
import torch.nn as nn

sample_data = torch.Tensor([1, 2, -1, -1])
myRelu = nn.ReLU()
print(myRelu(sample_data))
