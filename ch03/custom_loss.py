# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 19:44
# @Author  : jhys
# @FileName: custom_loss.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))


# 第二步：准备数据集，模拟一个线性拟合过程
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 将numpy数据转化为torch的张量
inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)

# 构建模型
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# 第三步： 构建模型，构建一个一层的网络模型
model = nn.Linear(input_size, output_size)

# 与模型相关的配置、损失函数、优化方式
# 使用自定义函数，等价于criterion = nn.MSELoss()

criterion = My_loss()

# 定义迭代优化算法， 使用的是随机梯度下降算法
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 第四步：训练模型
loss_history = []
# 第四步：训练模型，迭代训练
for epoch in range(num_epochs):
    #  前向传播计算网络结构的输出结果
    outputs = model(inputs)

    # 计算损失函数
    loss = criterion(outputs, targets)

    # 反向传播更新参数，三步策略，归零梯度——>反向传播——>更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息和保存loss
    loss_history.append(loss.item())
    if (epoch + 1) % 5 == 0:
        print('Epoch[{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 第五步：结果展示。画出原y与x的曲线与网络结构拟合后的曲线
predicted = model(torch.from_numpy(x_train)).detach().numpy()

plt.plot(x_train, y_train, 'ro', label="Original data")
plt.plot(x_train, predicted, label="Fitted line")
plt.legend()
plt.show()

# 画loss在迭代过程中的变化情况
plt.plot(loss_history, label="loss for every epoch")
plt.legend()
plt.show()
