#! /usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import numpy as np
import Auxiliary as A
import numpy as np
import matplotlib.pyplot as plt


'''
def generate_poisson(m):
    # print(m)
    k = 0
    p = np.exp(-m)
    # print('p', p)
    sum_poisson = p
    y = np.random.random()
    # print('y', y)
    if y < p:
        pass
    else:
        while y >= sum_poisson and p != 0:
            p = m * p / (k + 1)
            # print(p)
            sum_poisson += p
            # print('sum', sum_poisson)
            k += 1
        if y <= sum_poisson:
            return k - 1

m1 = []
i = 0

while i < 10000:
    a = generate_poisson(1)
    # print(a)
    if a != None:
        m1.append(a)
        i += 1

m = np.mean(m1)

c = np.cov(m1)

# fig = plt.figure(dpi=64, figsize=(16, 9))


'''
'''
def A(i):

    if i == 0 or i == 1:

        return 1

    else:

        tmp = 1

        for j in range(1, i + 1):

            tmp = tmp * j

        return tmp


def left(m, j):

        tmp = 0

        for j in range(0, j + 1):

            tmp = tmp + np.exp(-m) * pow(m, j) / A(j)

        return tmp


def right(m, j):

    tmp = 0

    for j in range(0, j + 2):

        tmp = tmp + np.exp(-m) * pow(m, j) / A(j)

    return tmp


m = 1

k = []

i = 0

y = np.random.random()

k_count = 0

while 1:

    if i > 9999:

        break

    if y < left(m, k_count):

        y = np.random.random()

        k_count = 0

    if left(m, k_count) <= y <= right(m, k_count):

        k.append(k_count)

        i += 1

        print(i, y, left(m, k_count), right(m, k_count))

        y = np.random.random()

        k_count = 0

        continue

    else:

        k_count += 1

mean = np.mean(k)

cov = np.cov(k)


def test(a, b):

    for i in range(a, b):

        print(left(m, i), '~', right(m, i))
'''

# path = '/Users/tyz/Desktop/DataSets'

path = 'C:/Users/iamtyz/Desktop/DataSets'

# out = Auxiliary.MatrixComparison(path, 0)

# p = '/Users/tyz/Desktop/DataSets/matrix/8000-m.txt'

# np.savetxt('/Users/tyz/Desktop/DataSets/1-m.txt', tmp)

