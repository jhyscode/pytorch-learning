# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 11:20
# @Author  : jhys
# @FileName: autograd_test2.py

import torch

#创建一个二元函数，即z=f(x,y)=x2+y2，x可求导，y设置不可求导
x=torch.tensor(3.0,requires_grad=True)
y=torch.tensor(4.0,requires_grad=False)
z=torch.pow(x,2)+torch.pow(y,2)

#判断x,y是否是可以求导的
print(x.requires_grad)
print(y.requires_grad)
print(z.requires_grad)

z.backward()

#查看导数，也即所谓的梯度
print(x.grad)
print(y.grad)