#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-8-6 下午11:13
# @Author  : rhys
# @Software: PyCharm
# @Project : tf_s


import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data*2.56 + 0.15

    Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([1]))

    y = Weights*x_data + biases

    loss = tf.reduce_mean(tf.square(y-y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for step in range(20000):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(Weights), sess.run(biases))