# Implement this for a project that: 生成用于图像识别的代码


import os
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.utils import column_or_1d
import matplotlib.pyplot as plt
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cPickle
import cv2
import time

# Define the flags here
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128,
                            )
tf.app.flags.DEFINE_integer('num_epochs', 10,
                            )
tf.app.flags.DEFINE_boolean('shuffle', True,
                            )
tf.app.flags.DEFINE_boolean('flip', True,
                            )
tf.app.flags.DEFINE_boolean('rotate', True,
                            )
tf.app.flags.DEFINE_boolean('crop', True,
                            )
tf.app.flags.DEFINE_boolean('distort', True,
                            )
tf.app.flags.DEFINE_boolean('augment', True,
                            )
tf.app.flags.DEFINE_integer('width', 64,
                            )
tf.app.flags.DEFINE_integer('height', 64,
                            )
tf.app.flags.DEFINE_integer('depth', 3,
                            """