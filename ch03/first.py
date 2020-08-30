# -*- coding: utf-8 -*-
# @Time    : 2020/7/18 20:03
# @Author  : jhys
# @FileName: first.py

import torch

#运算
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
y = torch.rand(5, 3)
# print(x + y)


# 将torch的Tensor转化为NumPy数组
a = torch.ones(5)
# print(a)
b = a.numpy()
#print(b)

# 看NumPy数组是如何改变里面的值的：
a.add_(1)
print(a)
print(b)

