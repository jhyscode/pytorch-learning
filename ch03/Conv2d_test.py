# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 10:56
# @Author  : jhys
# @FileName: Conv2d_test.py

import torch
from torch.autograd import Variable
##单位矩阵来模拟输入
input=torch.ones(1,1,5,5)
input=Variable(input)
print(input)
x=torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,groups=1)
out=x(input)
print(out)