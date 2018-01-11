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
from datetime import datetime
sys.path.append('../')
from types import SimpleNamespace as Namespace
from feature.TimeSeriesFeatureExtractor import TimeSeriesFeatureExtractor

def vector2onehot(num_classes, labels_dense):
    return np.eye(num_classes)[labels_dense]

import pandas as pd

"""Record the start time of the algorithm"""
startTime = datetime.now()

classificationNum = 2
path = '../../data/trials/'
dataFileNames = ['0a.json','0b.json']
labels = [0,1] #['I-1', 'IV-2']
resultFileName = 'simpleFeatures0a0b.npz' #'simpleFeatures2a2b01.npz'

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

"""layer = 1"""
# num_unit_layer_2 = i
# random.seed()
# random.shuffle(rand_data)
# # print(rand_data[0].tolist())
# X_rand = rand_data[:,1:10]
# y_rand = rand_data[:,10]
# # print('888888888888', X_rand, '---------', y_rand)
#
# heldout_len = int(len(X_rand)*0.8)
# x_train = X_rand[:heldout_len]
# y_train = y_rand[:heldout_len]
# x_test = X_rand[heldout_len:]
# y_test = y_rand[heldout_len:]
# batch_xs = np.float32(x_train)
# batch_ys = np.float32(vector2onehot(numClass, np.array(y_train).astype(int)))
#
# test_xs = np.float32(x_test)
# test_ys = np.float32(vector2onehot(numClass, np.array(y_test).astype(int)))
# num_unit_layer_2 = 11
# run(numClass, numFeature, num_unit_layer_1, num_unit_layer_2, num_unit_layer_3, batch_xs, batch_ys)


"""layer = 2"""

for i in range(9,19):
    numClass = 2
    numFeature = 9
    num_unit_layer_1 = numClass
    num_unit_layer_2 = 0
    num_unit_layer_3 = 0
    num_hidden_layer = 2
    num_unit_layer_2 = i
    random.seed()
    random.shuffle(rand_data)
    # print(rand_data[0].tolist())
    X_rand = rand_data[:,1:10]
    y_rand = rand_data[:,10]
    # print('888888888888', X_rand, '---------', y_rand)

    heldout_len = int(len(X_rand)*0.8)
    x_train = X_rand[:heldout_len]
    y_train = y_rand[:heldout_len]
    x_test = X_rand[heldout_len:]
    y_test = y_rand[heldout_len:]
    batch_xs = np.float32(x_train)
    batch_ys = np.float32(vector2onehot(numClass, np.array(y_train).astype(int)))

    test_xs = np.float32(x_test)
    test_ys = np.float32(vector2onehot(numClass, np.array(y_test).astype(int)))

    # run(numClass, numFeature, num_unit_layer_1, num_unit_layer_2, num_unit_layer_3, batch_xs, batch_ys)
    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    # print('num_unit_layer_2',num_unit_layer_2)
    # print(batch_xs)
    tf_batch_xs = tf.constant(batch_xs)
    tf_batch_ys = tf.constant(batch_ys)

    """1 Hidden Layer"""
    # x  = tf.placeholder(tf.float32, [None, numFeature])
    # W = tf.Variable(tf.zeros([numFeature,numClass]))
    # b = tf.Variable(tf.zeros([numClass]))
    # y = tf.nn.relu( tf.matmul(x, W) + b )

    """2 Hidden Layer"""
    x  = tf.placeholder(tf.float32, [None, numFeature])
    W = tf.Variable(tf.zeros([numFeature,num_unit_layer_2]))
    b = tf.Variable(tf.zeros([num_unit_layer_2]))
    y2 = tf.nn.relu( tf.matmul(x, W) + b )
    W2 = tf.Variable(tf.zeros([num_unit_layer_2,numClass]))
    b2 = tf.Variable(tf.zeros([numClass]))
    y = tf.nn.relu( tf.matmul(y2, W2) + b2 )

    """3 Hidden Layer"""
    # x  = tf.placeholder(tf.float32, [None, numFeature])
    # W = tf.Variable(tf.zeros([numFeature,num_unit_layer_2]))
    # b = tf.Variable(tf.zeros([num_unit_layer_2]))
    # y2 = tf.nn.relu( tf.matmul(x, W) + b )
    #
    # W2 = tf.Variable(tf.zeros([num_unit_layer_2,num_unit_layer_3]))
    # b2 = tf.Variable(tf.zeros([num_unit_layer_3]))
    # y3 = tf.nn.relu( tf.matmul(y2, W2) + b2 )
    #
    # W3 = tf.Variable(tf.zeros([num_unit_layer_3,numClass]))
    # b3 = tf.Variable(tf.zeros([numClass]))
    # y = tf.nn.relu( tf.matmul(y3, W3) + b3 )



    y_ = tf.placeholder(tf.float32, [None, numClass])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)





    # sess = tf.InteractiveSession(config = tf.ConfigProto(log_device_placement=True))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess = tf.InteractiveSession(config = config)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print(sess.run(tf_batch_xs))
    tf_batch_xs = sess.run(tf.get_session_handle(tf_batch_xs))
    tf_batch_ys = sess.run(tf.get_session_handle(tf_batch_ys))

    batch_num = 1000
    batch_size = int(len(batch_xs)/batch_num)
    for i in range(batch_num):
        # if i%100 == 0:
        #     print(i)
        # print(i)
        # index = i*batch_size
        # print(index, batch_size)
        #
        # tf_batch_x = tf.slice(tf_batch_xs, [index, 0], [batch_size, -1])
        # tf_batch_y = tf.slice(tf_batch_ys, [index, 0], [batch_size, -1])
        # tf_batch_x = sess.run(tf.get_session_handle(tf_batch_x))
        # tf_batch_y = sess.run(tf.get_session_handle(tf_batch_y))
        # print('x:', sess.run(tf_batch_x))
        # print('y:',sess.run(tf_batch_y))
        # sess.run(train_step, feed_dict={x: tf_batch_x, y_: tf_batch_y})
        sess.run(train_step, feed_dict={x: tf_batch_xs, y_: tf_batch_ys})

    """tf.argmax(input, axis=None, name=None, dimension=None)"""
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print('Training accuracy:' , sess.run(accuracy, {x: batch_xs, y_: batch_ys}))
    # print('Testing accuracy:' , sess.run(accuracy, {x: test_xs, y_: test_ys}))
    """number of layers, hidden layer 1, hidden layer 2, hidden layer 3, training accuracy, testing accuracy"""
    print(str(num_hidden_layer),str(num_unit_layer_2),str(num_unit_layer_3),str(sess.run(accuracy, {x: batch_xs, y_: batch_ys})),str(sess.run(accuracy, {x: test_xs, y_: test_ys})))
    sess.close()
    del x
    del W
    del b
    del y2
    del W2
    del b2
    del y
    del sess



# endTime = datetime.now()
# print('Running Time:',str(endTime - startTime),' ',str(startTime), str(endTime))











"""End of file"""
