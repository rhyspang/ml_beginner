# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       simple_linear_model
   Description: 
   Author:          rhys
   date:            17/8/9
-------------------------------------------------
"""


import tensorflow as tf


W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(y - linear_model))

optimizer = tf.train.GradientDescentOptimizer(.01)
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(1000):
        sess.run(train, {x: x_train, y: y_train})
        print(sess.run([W, b, loss], {x: x_train, y: y_train}))


