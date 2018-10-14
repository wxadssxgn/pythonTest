#! /usr/bin/env python
# -*- coding: utf-8 -*- 


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

x = np.linspace(0, 1, 51)
y = np.linspace(0, 1, 51)
data_x = []
data_y = []

for i in x:
    for j in y:
        data_x.append([i])
        data_y.append([j])

print(data_x, data_y)

x_in = tf.constant(data_x)
y_in = tf.constant(data_y)
phi = (1 / (tf.exp(np.pi) - tf.exp(-np.pi))) * tf.sin(np.pi * x_in) * (tf.exp(np.pi * y_in) - tf.exp(-np.pi * y_in))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = sess.run(x_in)
Y = sess.run(y_in)
Z = sess.run(phi)
ax.scatter(X, Y, Z, c='g')
plt.show()
