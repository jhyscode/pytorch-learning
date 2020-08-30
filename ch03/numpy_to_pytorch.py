# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 15:42
# @Author  : jhys
# @FileName: numpy_to_pytorch.py

import numpy as np
import torch

# 看改变NumPy数组是如何自动改变Torch张量的：
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

