# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 15:41
# @Author  : jhys
# @FileName: cuda_tensor_test.py

import torch

x = torch.randn(1)

if torch.cuda.is_available():
    device = torch.device("cuda")   # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype