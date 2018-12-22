#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import datetime as dt


class MLP(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, n_hidden_6, n_hidden_7, out_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), torch.nn.Dropout(0.0), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), torch.nn.Dropout(0.0), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), torch.nn.Dropout(0.0), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), torch.nn.Dropout(0.0), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), torch.nn.Dropout(0.0), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, n_hidden_6), torch.nn.Dropout(0.0), nn.ReLU(True))
        self.layer7 = nn.Sequential(nn.Linear(n_hidden_6, n_hidden_7), torch.nn.Dropout(0.0), nn.ReLU(True))
        self.layer8 = nn.Sequential(nn.Linear(n_hidden_7, out_dim))

    def forward(self, temp):
        temp = self.layer1(temp)
        temp = self.layer2(temp)
        temp = self.layer3(temp)
        temp = self.layer4(temp)
        temp = self.layer5(temp)
        temp = self.layer6(temp)
        temp = self.layer7(temp)
        temp = self.layer8(temp)
        return temp
