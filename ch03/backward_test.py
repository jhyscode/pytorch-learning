# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 20:23
# @Author  : jhys
# @FileName: backward_test.py

import torch
from torch.autograd import Variable

x = torch.Tensor([[1.,2.,3.],[4.,5.,6.]])
x = Variable(x, requires_grad=True)
y = x + 2
z = y*y*3
out = z.mean()
print(x)
print(y)
print(z)
print(out)

print(x.grad_fn)
print(y.grad_fn)
print(z.grad_fn)
print(out.grad_fn)

#若是关于graph leaves求导的结果变量是一个标量，那么gradient默认为None，或者指定为“torch.Tensor([1.0])”
#若是关于graph leaves求导的结果变量是一个向量，那么gradient是不能缺省的，要是和该向量同纬度的tensor

# out.backward()
# print(x.grad)

gradients = torch.Tensor([[1., 1., 1.],[1., 1., 1.]])
z.backward(gradient=gradients)
print(x.grad)