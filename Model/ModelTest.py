#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import datetime as dt
import matplotlib.pyplot as plt

import NetModel


in_dim, h1, h2, h3, h4, h5, h6, h7, out_dim = 2, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 501
model = NetModel.MLP(in_dim, h1, h2, h3, h4, h5, h6, h7, out_dim)
model.load_state_dict(torch.load('parameters.pt'))

model.eval()

testlabel = np.array([2.7, 2.9])
testlabel = torch.from_numpy(testlabel)
testlabel = testlabel.float()
testout = model(Variable(testlabel))
testout = testout.data
testout = testout.numpy()
print(testout)

x = np.linspace(10, 20, 501)
plt.scatter(x, testout)
plt.show()
