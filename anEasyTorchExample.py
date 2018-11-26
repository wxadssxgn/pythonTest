#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

a = nn.Conv2d(2, 3, 4, 1)
print(a)
