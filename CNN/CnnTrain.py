#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import datetime as dt
import random
import CnnModel
import Auxiliary

model = CnnModel.CNN()

model.load_state_dict(torch.load('CnnParameters.pt'))

# path = '/Users/tyz/Desktop/DataSets'

path = 'C:/Users/iamtyz/Desktop/DataSets'

batch = 15

# DataSets = list(range(1, 101))

DataSets = list(range(20000, 23835))

DataSetsOrders = Auxiliary.Shuffle(DataSets, batch)

criterion = nn.MSELoss(reduce=True, size_average=False)

optimizer = torch.optim.Adam(model.parameters())

epoch = 0

loop = 0

# for epoch in range(5000):

while 1:

    time = dt.datetime.now().isoformat()

    order = DataSetsOrders[loop: loop + batch]

    if loop == int(len(DataSetsOrders) / batch):

        loop = 0

    loop += batch

    epoch += 1

    # order = random.sample(DataSetsOrders, batch)

    a = Auxiliary.GetMatrix(path, order)

    b, c = Auxiliary.GetS11(path, order)

    d = np.zeros([batch, 49, 1])

    for j in range(batch):

        for i in range(np.size(b[0])):

            temp = abs(complex(b[j][i], c[j][i]))

            d[j][i] = temp

    a = torch.from_numpy(a)

    a = a.float()

    a = a.view(batch, 1, 16, 16)

    d = torch.from_numpy(d)

    d = d.float()

    d = d.view(batch, 49)

    x = Variable(a, requires_grad=False)

    y = Variable(d, requires_grad=False)

    netout = model(x)

    loss = criterion(netout, y)
    
    if epoch % 1 == 0:

        print(time, epoch, loss.data[0])

    if loss.data[0] < 5e-3:

        print(time, epoch, loss.data[0])

        break

    if epoch % 100 == 0:

        torch.save(model.state_dict(), 'CnnParameters.pt')

        model.load_state_dict(torch.load('CnnParameters.pt'))
    
    # print(time, epoch, loss.data[0])
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 2000 == 0:

        DataSetsOrders = Auxiliary.Shuffle(DataSets, batch)

        loop = 0

torch.save(model.state_dict(), 'CnnParameters.pt')
