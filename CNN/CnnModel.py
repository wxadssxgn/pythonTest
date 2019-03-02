#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import datetime as dt

class CNN(nn.Module):
    def __init__(self, in_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, \
                                   kernel_size=5, stride=1, padding=2), \
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2))
    