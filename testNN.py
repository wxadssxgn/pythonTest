#! /usr/bin/env python
# -*- coding: utf-8 -*- 


import tensorflow as tf
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


x = np.linspace(0, 1, 51)
y = np.linspace(0, 0.6, 31)
data_x = []
data_y = []
hidden = 80
learning_rate = 0.001
epoch = 10000

for i in x:
    for j in y:
        data_x.append([i])
        data_y.append([j])

print(data_x, data_y)

x_in = tf.constant(data_x)
y_in = tf.constant(data_y)

w0x = tf.Variable(tf.random_normal(shape=[1, hidden], stddev=1))
w0y = tf.Variable(tf.random_normal(shape=[1, hidden], stddev=1))
b0x = tf.Variable(tf.random_normal(shape=[1, hidden], stddev=1))
b0y = tf.Variable(tf.random_normal(shape=[1, hidden], stddev=1))
w1 = tf.Variable(tf.random_normal(shape=[hidden, 1], stddev=1))
b1 = tf.Variable(tf.random_normal(shape=[1, 1], stddev=1))

a = tf.sigmoid(tf.matmul(x_in, w0x) + b0x) + tf.sigmoid(tf.matmul(y_in, w0y) + b0y)
net_out = tf.matmul(a, w1) + b1

dsigmoidh = tf.sigmoid(a) * (1 - tf.sigmoid(a))
pd_x = w0x * (tf.transpose(w1) * dsigmoidh)
pd_y = w0y * (tf.transpose(w1) * dsigmoidh)
dsigmoidh2 = tf.sigmoid(a) + dsigmoidh - 2 * tf.sigmoid(a) * dsigmoidh
pd_x2 = (w0x * w0x) * (tf.transpose(w1) * dsigmoidh2)
pd_y2 = (w0y * w0y) * (tf.transpose(w1) * dsigmoidh2)

loss = tf.reduce_mean((x_in * (1 - x_in) * (y_in * (0.6 - y_in) * pd_y2 + (2 - 4 * y_in) * pd_y - 2 * net_out) + y_in *
                       (0.6 - y_in) * (x_in * (1 - x_in) * pd_x2 + (2 - 4 * x_in) * pd_x - 2 * net_out) + 10) ** 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize((loss))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(1, epoch + 1):
    sess.run(train)
    time = dt.datetime.now().isoformat()
    print(time, 'step:', step, 'loss:', sess.run(loss))


phi = x_in * (1 - x_in) * y_in * (0.6 - y_in) * net_out


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = sess.run(x_in)
Y = sess.run(y_in)
Z = sess.run(phi)
ax.scatter(X, Y, Z, c='g')
plt.show()
