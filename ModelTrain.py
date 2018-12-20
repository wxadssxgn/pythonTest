#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import datetime as dt

import txtRead
import NetModel

path = "C:/Users/iamtyz/Desktop/data"

filepath, filelist = txtRead.getfilename(path)
filedata = txtRead.getdata(filepath)
filelabel = []
filedata = filedata[:, :, 1]
for i in range(np.size(filelist)):
    filelabel.append([filelist[i].split('-')[0], filelist[i].split('-')[1]])

filelabel = np.array(filelabel, dtype=float)

data = torch.from_numpy(filedata)
data = data.float()
label = torch.from_numpy(filelabel)
label = label.float()

in_dim, h1, h2, h3, h4, h5, h6, h7, out_dim = 2, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 501
model = NetModel.MLP(in_dim, h1, h2, h3, h4, h5, h6, h7, out_dim)

x = Variable(label, requires_grad=False)
y = Variable(data, requires_grad=False)

criterion = nn.MSELoss(size_average=False)
# optimizer = optim.SGD(model.parameters(), lr=1e-5)
optimizer = torch.optim.Adam(model.parameters())

t = 0

while 1:
    time = dt.datetime.now().isoformat()
    t = t + 1
    netout = model(x)
    loss = criterion(netout, y)
    print(time, t, loss.data[0])
    if loss.data[0] < 0.001:
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.save(model.state_dict(), 'para.pt')
