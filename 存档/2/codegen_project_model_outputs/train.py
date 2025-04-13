# Implement this for a project that: 生成用于图像识别的代码


# 参考了一下https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 一个简单的模型
# 先把数据维度降为1，然后把它拉平成一个数字的一维向量
# 这样就可以用数字识别一个数字了
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 输入层
# 将输入的28*28的图像数据拉平成一个784维的数据
# 这里的None是因为没有什么模型，只是定义了一个空的图像
# 可以在模型中看到，这里的None是因为有一个输入，它是一个占位符，所以不会�

from model import *
from dataset import *
