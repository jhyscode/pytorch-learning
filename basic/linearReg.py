# -*- coding: utf-8 -*-
# @Time    : 2020/8/26 20:36
# @Author  : jhys
# @FileName: linearReg.py

import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)  #前向传播
    loss = criterion(y_pred, y_data) #计算loss

    if (epoch % 5 == 0):
        print(epoch, loss)

    optimizer.zero_grad()
    loss.backward()  #反向传播
    optimizer.step() #更新参数


# Output weight and bias
print('w = ',model.linear.weight.item())
print('b = ',model.linear.bias.item())

# Test Model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pred = ", y_test.data)


