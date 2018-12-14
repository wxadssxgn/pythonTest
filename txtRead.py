# ! /usr/bin/env python
# -*- coding: utf-8 -*- 

import os


def eachFile(filepath):
    pathDir = os.listdir(filepath)
    for s in pathDir:
        newDir = os.path.join(filepath, s)
        if os.path.isfile(newDir):
            if os.path.splitext(newDir)[1] == ".txt":
                os.readFile(newDir)
                pass
        else:
            eachFile(newDir)
