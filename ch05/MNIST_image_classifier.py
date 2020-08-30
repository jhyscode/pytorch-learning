# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 16:23
# @Author  : jhys
# @FileName: MNIST_image_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data

is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
sample_data = next(iter(train_loader))


def plot_img(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image, cmap='gray')
    plt.show()


plot_img(sample_data[0][2])
plot_img(sample_data[0][1])

