# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 13:00
# @Author  : jhys
# @FileName: autograd_test4.py

import torch
'''
x 是一个（2,3）的矩阵，设置为可导，是叶节点，即leaf variable
y 是中间变量,由于x可导，所以y可导
z 是中间变量,由于x，y可导，所以z可导
f 是一个求和函数，最终得到的是一个标量scaler
'''

x = torch.tensor([[1.,2.,3.],[4.,5.,6.]],requires_grad=True)
y = torch.add(x,1)
z = 2*torch.pow(y,2)
f = torch.mean(z)

print(x.requires_grad)
print(y.requires_grad)
print(z.requires_grad)
print(f.requires_grad)
print('===================================')
f.backward()
print(x.grad)