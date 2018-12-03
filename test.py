#! /usr/bin/env python
# -*- coding: utf-8 -*- 


import tensorflow as tf
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
'''
x = np.random.rand(400, 1)
y = np.random.rand(400, 1)
'''
data_x = []
data_y = []
hidden = 4000
learning_rate = 1e-3

for i in x:
    for j in y:
        data_x.append([i])
        data_y.append([j])

x_in = tf.constant(data_x, dtype=tf.float32)
y_in = tf.constant(data_y, dtype=tf.float32)

w0x = tf.Variable(tf.random_normal(shape=[1, hidden], stddev=1), dtype=tf.float32)
w0y = tf.Variable(tf.random_normal(shape=[1, hidden], stddev=1), dtype=tf.float32)
b0x = tf.Variable(tf.random_normal(shape=[1, hidden], stddev=1), dtype=tf.float32)
b0y = tf.Variable(tf.random_normal(shape=[1, hidden], stddev=1), dtype=tf.float32)
w1 = tf.Variable(tf.random_normal(shape=[hidden, 1], stddev=1), dtype=tf.float32)
b1 = tf.Variable(tf.random_normal(shape=[1, 1], stddev=1), dtype=tf.float32)

a = tf.sigmoid(tf.matmul(x_in, w0x) + b0x) + tf.sigmoid(tf.matmul(y_in, w0y) + b0y)
net_out = tf.matmul(a, w1) + b1

dsigmoidh = tf.sigmoid(a) * (1 - tf.sigmoid(a))
pd_x = w0x * (tf.transpose(w1) * dsigmoidh)
pd_y = w0y * (tf.transpose(w1) * dsigmoidh)
dsigmoidh2 = tf.sigmoid(a) + dsigmoidh - 2 * tf.sigmoid(a) * dsigmoidh
pd_x2 = (w0x * w0x) * (tf.transpose(w1) * dsigmoidh2)
pd_y2 = (w0y * w0y) * (tf.transpose(w1) * dsigmoidh2)

# global_step=tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(1.0, global_step, 1000, 0.96, staircase=False)

tral = tf.abs(x_in * (1 - x_in) * (y_in * (1 - y_in) * pd_y2 + (2 - 4 * y_in) * pd_y - 2 * net_out) + y_in *
                       (1 - y_in) * (x_in * (1 - x_in) * pd_x2 + (2 - 4 * x_in) * pd_x - 2 * net_out) - (np.pi ** 2) *
                       y_in * tf.sin(np.pi * x_in))
loss = tf.reduce_mean(tral)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize((loss))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
'''
for step in range(1, epoch + 1):
    sess.run(train)
    time = dt.datetime.now().isoformat()
    print(time, 'step:', step, 'loss:', sess.run(loss))
'''
sess.run(train)
step = 1
time = dt.datetime.now().isoformat()
print(time, 'step:', step, 'loss:', sess.run(loss))
temp = sess.run(loss)

while sess.run(loss):
    sess.run(train)
    step = step + 1
    time = dt.datetime.now().isoformat()
    print(time, 'step:', step, 'loss:', sess.run(loss))
    if abs(sess.run(loss) - temp) < 0.00000001:
        break
    else:
        temp = sess.run(loss)


phi_t = x_in * (1 - x_in) * y_in * (1 - y_in) * net_out + y_in * tf.sin(np.pi * x_in)
Z_t = sess.run(phi_t)


fig_t = plt.figure()
Y_t = sess.run(y_in)
ax = fig_t.add_subplot(111, projection='3d')

X_t = sess.run(x_in)
Z_t = sess.run(phi_t)
ax.scatter(X_t, Y_t, Z_t, c='k')
plt.show()

sess.close()

#----------------------------------------------------------------------------------
'''
data_x = []
data_y = []


for i in x:
    for j in y:
        data_x.append([i])
        data_y.append([j])

x_in = tf.constant(data_x)
y_in = tf.constant(data_y)
phi_a = (1 / (tf.exp(np.pi) - tf.exp(-np.pi))) * tf.sin(np.pi * x_in) * (tf.exp(np.pi * y_in) - tf.exp(-np.pi * y_in))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = sess.run(x_in)
Y = sess.run(y_in)
Z = -(sess.run(phi_a) - Z_t)
ax.scatter(X, Y, Z, c='g')
plt.show()

sess.close()
print(max(abs(Z)))
'''