#! /usr/bin/env python
# -*- coding: utf-8 -*- 


import tensorflow as tf
import numpy as np
import datetime as dt

x_in = tf.Variable(tf.random_uniform(shape=[20, 1], minval=0.0, maxval=1.0), dtype=tf.float32, trainable=False)
y_in = tf.Variable(tf.random_uniform(shape=[20, 1], minval=0.0, maxval=1.0), dtype=tf.float32, trainable=False)

hidden = 500
learning_rate = 1e-4
# global_step=tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(1e-2, global_step, 2e4, 0.96, staircase=False)

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


tral = tf.abs(x_in * (1 - x_in) * (y_in * (1 - y_in) * pd_y2 + (2 - 4 * y_in) * pd_y - 2 * net_out) + y_in *
                       (1 - y_in) * (x_in * (1 - x_in) * pd_x2 + (2 - 4 * x_in) * pd_x - 2 * net_out) - (np.pi ** 2) *
                       y_in * tf.sin(np.pi * x_in))
loss = tf.reduce_mean(tral)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize((loss))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

step = 0

while 1:
    step = step + 1
    sess.run(train)
    if step % 2000 == 0:
        time = dt.datetime.now().isoformat()
        print(time, 'step:', step, 'loss:', sess.run(loss))
    if sess.run(loss) < 0.1:
        print(time, 'step:', step, 'loss:', sess.run(loss))
        break

sess.close()

'''
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
'''