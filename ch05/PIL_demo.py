# -*- coding: utf-8 -*-
# @Time    : 2020/8/2 19:37
# @Author  : jhys
# @FileName: PIL_demo.py

import torch
from torchvision import transforms
import cv2
import numpy as np

img_path = '../ch05/Emperor_penguins.jpg'

transform1 = transforms.Compose([
    transforms.ToTensor(),
])
##numpy.ndarray
img = cv2.imread(img_path)# 读取图像
img1 = transform1(img)
print("img1 = ",img1)

# 转化为numpy.ndarray并显示
img_1 = img1.numpy()*255
img_1 = img_1.astype('uint8')
img_1 = np.transpose(img_1, (1,2,0))
cv2.imshow('img_1', img_1)
cv2.waitKey()

