#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
'''
class MLP(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2,n_hidden_3 , out_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))

        def forward(self, temp):
            temp = self.layer1(temp)
            temp = self.layer2(temp)
            temp = self.layer3(temp)
            temp = self.layer4(temp)
            return temp

learning_rate = 1e-2
epoch = 200

net = MLP(2, 50, 50, 50, 1)
criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)

while loss.data[0] > 1:
    data = np.random.rand(50, 2)
    net_in = Variable(data, requires_grad=False)
    opimizer.zero_grad()
    net_out = net(net_in)
    
    loss = criterion(net_out, 0)
'''
'''
net_in = Variable(data)
net_out = net(net_in)
target = 0
criterion = nn.MSELoss()
loss = criterion(net_out, target)
net.zero_grad()
'''
dtype = torch.FloatTensor
N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
learning_rate = 1e-6
for t in range(5000):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])
    loss.backward()
    w1.data -= learning_rate * w1.grad.data
    w2.data-= learning_rate * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
