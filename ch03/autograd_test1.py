# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 10:54
# @Author  : jhys
# @FileName: autograd_test1.py

import torch

x = torch.tensor(3.0, requires_grad=True)
y = torch.pow(x, 2)

print(x.requires_grad)
print(y.requires_grad)

#求导，通过backward函数来实现
y.backward()

#查看导数，也即所谓的梯度
print(x.grad)