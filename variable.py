#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-8-7 下午9:16
# @Author  : rhys
# @Software: PyCharm
# @Project : tf_s

import tensorflow as tf


def main():
    state = tf.Variable(0, name='counter')
    one = tf.constant(1)

    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            print(sess.run(update))
            print(sess.run(state))

if __name__ == '__main__':
    main()