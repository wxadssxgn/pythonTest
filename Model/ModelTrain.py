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

dtype = torch.cuda.FloatTensor
# dtype = torch.FloatTensor

path = "C:/Users/wxadssxgn/Desktop/aaa"

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

model = NetModel.MLP(in_dim, h1, h2, h3, h4, h5, h6, h7, out_dim).cuda()

model.load_state_dict(torch.load('parameters.pt'))

x = Variable(label.type(dtype), requires_grad=False)
y = Variable(data.type(dtype), requires_grad=False)

criterion = nn.MSELoss(reduce=True, size_average=False)
# optimizer = optim.SGD(model.parameters(), lr=1e-5)
optimizer = torch.optim.Adam(model.parameters())

t = 0

while 1:
    time = dt.datetime.now().isoformat()
    t = t + 1
    netout = model(x)
    loss = criterion(netout, y)
    if t % 1 == 0:
        print(time, t, loss.data)
    if loss.data < 0.0001:
        print(time, t, loss.data)
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'parameters.pt')
