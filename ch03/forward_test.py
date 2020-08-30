# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 20:09
# @Author  : jhys
# @FileName: forward_test.py

import torch

input = torch.ones([2, 2], requires_grad=False)
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(3.0, requires_grad=True)
w3 = torch.tensor(4.0, requires_grad=True)

l1 = input * w1
l2 = l1 + w2
l3 = l1 * w3
l4 = l2 * l3
loss = l4.mean()

print(w1.data, w1.grad, w1.grad_fn)

print(l1.data, l1.grad, l1.grad_fn)

print(loss.data, loss.grad, loss.grad_fn)

