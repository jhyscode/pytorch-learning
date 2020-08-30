# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 16:00
# @Author  : jhys
# @FileName: autograd_test.py

import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 3
out = z.mean()
print(out)
print(z, out)

out.backward()
print(x.grad)
