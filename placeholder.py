#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-8-8 下午10:57
# @Author  : rhys
# @Software: PyCharm
# @Project : tf_s

import tensorflow as tf


def main():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1: [2], input2: [9]}))

if __name__ == '__main__':
    main()