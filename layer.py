#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-8-8 下午11:11
# @Author  : rhys
# @Software: PyCharm
# @Project : tf_s

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function:
        return activation_function(Wx_plus_b)
    return Wx_plus_b


def main():
    x_data = np.linspace(-2, 2, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) + noise

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    prediction = add_layer(l1, 10, 1, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(.1).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_data, y_data)
        plt.ion()
        plt.show()

        for i in range(1000):
                sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

                # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

                try:
                    ax.lines.remove(lines[0])
                except Exception as e:
                    pass
                prediction_value = sess.run(prediction, feed_dict={xs: x_data})
                lines = ax.plot(x_data, prediction_value, 'r-', lw=1)
                plt.pause(0.01)

if __name__ == '__main__':
    main()
