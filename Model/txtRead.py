# ! /usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import numpy as np


def getfilename(path):
    filelist = []
    filepath = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            tempfilename = path + '/' + filename
            filepath.append(tempfilename)
            # filelist.append(filename)
            filelist.append(filename.split('.t')[0])
    return filepath, filelist


def getdata(filepath):
    data = []
    numfile = np.size(filepath)
    for i in range(numfile):
        f = open(filepath[i])
        tempdata = []
        line = f.readline()
        while line:
            num = list(map(float, line.split()))
            tempdata.append(num)
            line = f.readline()
        f.close()
        data.append(tempdata)
    data = np.array(data)
    return data
