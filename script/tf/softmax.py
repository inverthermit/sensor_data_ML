from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace
from feature.TimeSeriesFeatureExtractor import TimeSeriesFeatureExtractor

def vector2onehot(num_classes, labels_dense):
    return np.eye(num_classes)[labels_dense]

import pandas as pd

classificationNum = 3
path = '../../data/AUSMELT-71/'
dataFileNames = [#'Trial 2A 21.12.2017.json','Trial 2B 21.12.2017.json',
 'drain.json','Trial 2A 21.12.2017.json','Pin hole tip.json','Trial 3B 22.12.2017.json', 'Trial 3C 22.12.2017.json']
labels = [#0, 1,
            0,0,1,1,2] #['I-1', 'IV-2']
resultFileName = 'simpleFeatures2a2b3a3b3c01021.npz' #'simpleFeatures2a2b01.npz'

extractor = TimeSeriesFeatureExtractor()
extractor.saveSimpleFeaturedData(path, dataFileNames, labels, resultFileName)

fileContent = np.load(path + resultFileName)
data = fileContent['data']

"""['timeStamp',
    'x', 'y', 'z',
    'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z',
    'Rolling_Std_x','Rolling_Std_y','Rolling_Std_z',
    'label']
"""
train_no_nan = extractor.insertRollingFeatures(data, window = 350)

print('****************Start to run classifications***************')
rand_data = np.array(copy.deepcopy(train_no_nan))
random.shuffle(rand_data)
print(rand_data[0].tolist())
X_rand = rand_data[:,1:10]
y_rand = rand_data[:,10]
# print('888888888888', X_rand, '---------', y_rand)

heldout_len = int(len(X_rand)*0.8)
x_train = X_rand[:heldout_len]
y_train = y_rand[:heldout_len]
x_test = X_rand[heldout_len:]
y_test = y_rand[heldout_len:]


# batch_xs = [[-0.3301766753196716, -0.19591960310935974, -0.14742934703826904, -0.9847284739358084, -0.11343124378173212, -0.023986388131244374, 0.09911378051724326, 0.14364807808884125, 0.15149487064158973],
# [-0.9301766753196716, -0.19591960310935974, -0.14742934703826904, -0.9847284739358084, -0.11343124378173212, -0.023986388131244374, 0.09911378051724326, 0.14364807808884125, 0.15149487064158973],
# [-0.2301766753196716, -0.19591960310935974, -0.14742934703826904, -0.9847284739358084, -0.11343124378173212, -0.023986388131244374, 0.09911378051724326, 0.14364807808884125, 0.15149487064158973]]
#
# batch_ys = vector2onehot(3, [0,1,2])

batch_xs = x_train
batch_ys = vector2onehot(3, np.array(y_train).astype(int))

test_xs = x_test
test_ys = vector2onehot(3, np.array(y_test).astype(int))

numClass = 3
numFeature = 9

sess = tf.Session()

x  = tf.placeholder(tf.float32, [None, numFeature])
W = tf.Variable(tf.zeros([numFeature,numClass]))
b = tf.Variable(tf.zeros([numClass]))

y = tf.nn.relu( tf.matmul(x, W) + b )

y_ = tf.placeholder(tf.float32, [None, numClass])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    if i%1000 == 0:
        print(i)
    # print(i)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

"""tf.argmax(input, axis=None, name=None, dimension=None)"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Training accuracy:' , sess.run(accuracy, {x: batch_xs, y_: batch_ys}))
print('Testing accuracy:' , sess.run(accuracy, {x: test_xs, y_: test_ys}))














"""End of file"""
