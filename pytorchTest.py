#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

def buildFullyConnectedResidualNetwork(nBlocks, nNeurons,noutputs, ninputs):
    model = nn.Sequential()
    model:add(nn.Reshape(ninputs))
    