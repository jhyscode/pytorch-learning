# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 15:54
# @Author  : jhys
# @FileName: Sequential_test.py

# 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
import torch.nn as nn
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 20, 5)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(20, 64, 5)),
    ('relu2', nn.ReLU())
]))

print(model)
print(model[2])
