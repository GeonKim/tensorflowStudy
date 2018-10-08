#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 20:06:54 2018

@author: pirl
"""

import tensorflow as tf

#print tensor : 노드와 세션을 만들고 런
hello = tf.constant("hello, tf!")
sess = tf.Session()

print(sess.run(hello))

#add tensor
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.5)
node3 = tf.add(node1, node2)

print(sess.run(node3))


phNode1 = tf.placeholder(tf.float32)
phNode2 = tf.placeholder(tf.float32)
addNode = phNode1 + phNode2

print(sess.run(addNode, feed_dict={phNode1:10, phNode2:1.4}))
