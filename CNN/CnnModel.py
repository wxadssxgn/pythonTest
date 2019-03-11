#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import datetime as dt


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1, padding=0))

        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1, padding=0))

        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        self.layer6 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer7 = nn.Sequential(nn.Linear(128*7*7, 4096), torch.nn.Dropout(0.0), nn.ReLU(inplace=True))

        self.layer8 = nn.Sequential(nn.Linear(4096, 1024), torch.nn.Dropout(0.0), nn.ReLU(inplace=True))

        self.layer9 = nn.Sequential(nn.Linear(1024, 512), torch.nn.Dropout(0.0), nn.ReLU(inplace=True))

        self.layer10 = nn.Sequential(nn.Linear(512, 49))
    
    def forward(self, temp):

        temp = self.layer1(temp)

        temp = self.layer2(temp)

        temp = self.layer3(temp)

        temp = self.layer4(temp)

        temp = self.layer5(temp)

        temp = self.layer6(temp)

        temp = temp.view(temp.size(0), -1)

        temp = self.layer7(temp)

        temp = self.layer8(temp)

        temp = self.layer9(temp)

        temp = self.layer10(temp)

        return temp
