# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 12:53
# @Author  : jhys
# @FileName: autograd_test3.py

import torch

#创建一个多元函数，即Y=XW+b=Y=x1*w1+x2*w2*x3*w3+b，x不可求导，W,b设置可求导
X=torch.tensor([1.5,2.5,3.5],requires_grad=False)
W=torch.tensor([0.2,0.4,0.6],requires_grad=True)
b=torch.tensor(0.1,requires_grad=True)
Y=torch.add(torch.dot(X,W),b)


#判断每个tensor是否是可以求导的
print(X.requires_grad)
print(W.requires_grad)
print(b.requires_grad)
print(Y.requires_grad)

Y.backward()


print(W.grad)
print(b.grad)