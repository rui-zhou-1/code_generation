# Implement this for a project that: 生成用于图像识别的代码


import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import cPickle
import scipy.misc

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import matplotlib.pyplot as plt

# from keras.regularizers import l2
# from keras.regularizers import l1
# from keras.regularizers import l1l2

# from keras.constraints import maxnorm
# from keras.constraints import nonneg

# from keras.layers.normalization import BatchNormalization
# from keras.layers.normalization import BatchNormalization
# from keras.layers.normalization import BatchNormalization

# from keras.layers.advanced_activations import LeakyReLU, PReLU

# from keras.layers.advanced_activations import ELU

# from keras.layers.advanced_activations import ThresholdedReLU

# from keras.layers.advanced_activations import Softmax

# from keras.layers.advanced_activations import Thresholded