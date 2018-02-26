# -*- coding: utf-8 -*-


from __future__ import division
import os
import numpy as np
from os.path import isfile, join
from os import listdir
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace
# from SimpleNamespace import SimpleNamespace as Namespace
import random
import os.path

from sklearn.cross_validation import train_test_split

from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau
import keras.utils.np_utils as kutils

import random
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from util.Util import Util

from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD

sys.path.append('../')
classificationNum = 3
rootDir = '../../'
path = rootDir + Util.getConfig('trials_folder_path')
tmpPath = rootDir + Util.getConfig('tmp_path')
savePath = rootDir + Util.getConfig('pic_to_np_array')
width = 80
height = 60

labels = {'1a.json':0,
	  '1b.json':0,
	  '1c.json':0,
	  '1d.json':0,
	  '1e.json':0,
	  '2a.json':0,
	  '2b.json':1,
	  '3a.json':0,
	  '3b.json':1,
	  '3c.json':1,
	  '3d.json':1,
	  '3e.json':1,
	  '3f.json':1,
	  '3g.json':1,
	  '4a.json':0,
	  '4b.json':1,
	  '4c.json':1,
	  '4d.json':1,
	  '4e.json':1,
	  '4f.json':1,
	  '4g.json':1,
	  '4h.json':0}

fileNames = [f for f in listdir(savePath) if isfile(join(savePath, f))]
train_file_names = random.sample(fileNames,17)
test_file_names = [x for x in fileNames if x not in train_file_names]
train_data_total = None
train_label_total = None
for file_name in train_file_names:
    train_data = np.load(savePath + file_name)
    train_label = None
    label = None

    # Getting the lable
    if labels[file_name.split('.')[0] + '.json'] == 0:
        label = np.array(([1,0]))
    else:
        label = np.array(([0,1]))

    for index in range(train_data['data'].shape[0]):
            if index == 0:
                    train_label = label
            else:
                    train_label = np.vstack((train_label, label))
    if train_data_total is None:
        train_data_total = train_data['data']
    else:
        train_data_total = np.vstack((train_data_total, train_data['data']))
    if train_label_total is None:
        train_label_total = train_label
    else:
        train_label_total = np.vstack((train_label_total, train_label))


test_data_total = None
test_label_total = None
for file_name in test_file_names:
    train_data = np.load(savePath + file_name)
    train_label = None
    label = None
    if labels[file_name.split('.')[0] + '.json'] == 0:
        label = np.array(([1,0]))
    else:
        label = np.array(([0,1]))
    for index in range(train_data['data'].shape[0]):
            if index == 0:
                    train_label = label
            else:
                    train_label = np.vstack((train_label, label))
    if test_data_total is None:
        test_data_total = train_data['data']
    else:
        test_data_total = np.vstack((test_data_total, train_data['data']))
    if test_label_total is None:
        test_label_total = train_label
    else:
        test_label_total = np.vstack((test_label_total, train_label))


epochs = 30
batch_size = 80
img_height, img_width = 64, 48

train_feature = train_data_total
test_feature = test_data_total
train_label = train_label_total
test_label = test_label_total
corrLabel = list()
corrLabel_test = list()
for i in range(train_feature.shape[0]):
	corrLabel.append((train_feature[i,:], train_label[i,:]))
for i in range(test_feature.shape[0]):
	corrLabel_test.append((test_feature[i,:], test_label[i,:]))

random.shuffle(corrLabel)
random.shuffle(corrLabel_test)
trainX= np.array([i[0] for i in corrLabel]).reshape(-1,img_width, img_height,1)
valX = np.array([i[0] for i in corrLabel_test]).reshape(-1,img_width, img_height,1)
trainY = np.array([i[1] for i in corrLabel]).reshape(-1, 2)
valY = np.array([i[1] for i in corrLabel_test]).reshape(-1, 2)

# trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.3)

print(trainX.shape, valX.shape, trainY.shape, valY.shape)

# input_shape = (img_width, img_height, 1)
# input_shape = Input(shape=(3, img_width, img_height))

# input = Input(shape=(1, img_width, img_height))

conv1_7x7_s2 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1/7x7_s2',W_regularizer=l2(0.0002))(Input(shape=(3, img_width, img_height)))

conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

# pool1_helper = PoolHelper()(conv1_zero_pad)

pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool1/3x3_s2')(conv1_zero_pad)

# pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

conv2_3x3_reduce = Convolution2D(64,1,1,border_mode='same',activation='relu',name='conv2/3x3_reduce',W_regularizer=l2(0.0002))(pool1_3x3_s2)

conv2_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='conv2/3x3',W_regularizer=l2(0.0002))(conv2_3x3_reduce)

# conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)

conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_3x3)

# pool2_helper = PoolHelper()(conv2_zero_pad)

pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool2/3x3_s2')(conv2_zero_pad)


inception_3a_1x1 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3a/1x1',W_regularizer=l2(0.0002))(pool2_3x3_s2)

inception_3a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_3a/3x3_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)

inception_3a_3x3 = Convolution2D(128,3,3,border_mode='same',activation='relu',name='inception_3a/3x3',W_regularizer=l2(0.0002))(inception_3a_3x3_reduce)

inception_3a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_3a/5x5_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)

inception_3a_5x5 = Convolution2D(32,5,5,border_mode='same',activation='relu',name='inception_3a/5x5',W_regularizer=l2(0.0002))(inception_3a_5x5_reduce)

inception_3a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3a/pool')(pool2_3x3_s2)

inception_3a_pool_proj = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3a/pool_proj',W_regularizer=l2(0.0002))(inception_3a_pool)

inception_3a_output = merge([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj],mode='concat',concat_axis=1,name='inception_3a/output')


inception_3b_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/1x1',W_regularizer=l2(0.0002))(inception_3a_output)

inception_3b_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/3x3_reduce',W_regularizer=l2(0.0002))(inception_3a_output)

inception_3b_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='inception_3b/3x3',W_regularizer=l2(0.0002))(inception_3b_3x3_reduce)

inception_3b_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3b/5x5_reduce',W_regularizer=l2(0.0002))(inception_3a_output)

inception_3b_5x5 = Convolution2D(96,5,5,border_mode='same',activation='relu',name='inception_3b/5x5',W_regularizer=l2(0.0002))(inception_3b_5x5_reduce)

inception_3b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3b/pool')(inception_3a_output)

inception_3b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3b/pool_proj',W_regularizer=l2(0.0002))(inception_3b_pool)

inception_3b_output = merge([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj],mode='concat',concat_axis=1,name='inception_3b/output')


inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)

# pool3_helper = PoolHelper()(inception_3b_output_zero_pad)

pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool3/3x3_s2')(inception_3b_output_zero_pad)


inception_4a_1x1 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='inception_4a/1x1',W_regularizer=l2(0.0002))(pool3_3x3_s2)

inception_4a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_4a/3x3_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)

inception_4a_3x3 = Convolution2D(208,3,3,border_mode='same',activation='relu',name='inception_4a/3x3',W_regularizer=l2(0.0002))(inception_4a_3x3_reduce)

inception_4a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_4a/5x5_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)

inception_4a_5x5 = Convolution2D(48,5,5,border_mode='same',activation='relu',name='inception_4a/5x5',W_regularizer=l2(0.0002))(inception_4a_5x5_reduce)

inception_4a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4a/pool')(pool3_3x3_s2)

inception_4a_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4a/pool_proj',W_regularizer=l2(0.0002))(inception_4a_pool)

inception_4a_output = merge([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj],mode='concat',concat_axis=1,name='inception_4a/output')


loss1_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss1/ave_pool')(inception_4a_output)

loss1_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss1/conv',W_regularizer=l2(0.0002))(loss1_ave_pool)

loss1_flat = Flatten()(loss1_conv)

loss1_fc = Dense(1024,activation='relu',name='loss1/fc',W_regularizer=l2(0.0002))(loss1_flat)

loss1_drop_fc = Dropout(0.7)(loss1_fc)

loss1_classifier = Dense(1000,name='loss1/classifier',W_regularizer=l2(0.0002))(loss1_drop_fc)

loss1_classifier_act = Activation('softmax')(loss1_classifier)


inception_4b_1x1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4b/1x1',W_regularizer=l2(0.0002))(inception_4a_output)

inception_4b_3x3_reduce = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4b/3x3_reduce',W_regularizer=l2(0.0002))(inception_4a_output)

inception_4b_3x3 = Convolution2D(224,3,3,border_mode='same',activation='relu',name='inception_4b/3x3',W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)

inception_4b_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4b/5x5_reduce',W_regularizer=l2(0.0002))(inception_4a_output)

inception_4b_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4b/5x5',W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)

inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4b/pool')(inception_4a_output)

inception_4b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4b/pool_proj',W_regularizer=l2(0.0002))(inception_4b_pool)

inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],mode='concat',concat_axis=1,name='inception_4b_output')


inception_4c_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/1x1',W_regularizer=l2(0.0002))(inception_4b_output)

inception_4c_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/3x3_reduce',W_regularizer=l2(0.0002))(inception_4b_output)

inception_4c_3x3 = Convolution2D(256,3,3,border_mode='same',activation='relu',name='inception_4c/3x3',W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)

inception_4c_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4c/5x5_reduce',W_regularizer=l2(0.0002))(inception_4b_output)

inception_4c_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4c/5x5',W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)

inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4c/pool')(inception_4b_output)

inception_4c_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4c/pool_proj',W_regularizer=l2(0.0002))(inception_4c_pool)

inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],mode='concat',concat_axis=1,name='inception_4c/output')


inception_4d_1x1 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4d/1x1',W_regularizer=l2(0.0002))(inception_4c_output)

inception_4d_3x3_reduce = Convolution2D(144,1,1,border_mode='same',activation='relu',name='inception_4d/3x3_reduce',W_regularizer=l2(0.0002))(inception_4c_output)

inception_4d_3x3 = Convolution2D(288,3,3,border_mode='same',activation='relu',name='inception_4d/3x3',W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)

inception_4d_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4d/5x5_reduce',W_regularizer=l2(0.0002))(inception_4c_output)

inception_4d_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4d/5x5',W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)

inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4d/pool')(inception_4c_output)

inception_4d_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4d/pool_proj',W_regularizer=l2(0.0002))(inception_4d_pool)

inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],mode='concat',concat_axis=1,name='inception_4d/output')


loss2_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss2/ave_pool')(inception_4d_output)

loss2_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss2/conv',W_regularizer=l2(0.0002))(loss2_ave_pool)

loss2_flat = Flatten()(loss2_conv)

loss2_fc = Dense(1024,activation='relu',name='loss2/fc',W_regularizer=l2(0.0002))(loss2_flat)

loss2_drop_fc = Dropout(0.7)(loss2_fc)

loss2_classifier = Dense(1000,name='loss2/classifier',W_regularizer=l2(0.0002))(loss2_drop_fc)

loss2_classifier_act = Activation('softmax')(loss2_classifier)


inception_4e_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_4e/1x1',W_regularizer=l2(0.0002))(inception_4d_output)

inception_4e_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4e/3x3_reduce',W_regularizer=l2(0.0002))(inception_4d_output)

inception_4e_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_4e/3x3',W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)

inception_4e_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4e/5x5_reduce',W_regularizer=l2(0.0002))(inception_4d_output)

inception_4e_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_4e/5x5',W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)

inception_4e_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4e/pool')(inception_4d_output)

inception_4e_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4e/pool_proj',W_regularizer=l2(0.0002))(inception_4e_pool)

inception_4e_output = merge([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj],mode='concat',concat_axis=1,name='inception_4e/output')


inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)

pool4_helper = PoolHelper()(inception_4e_output_zero_pad)

pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool4/3x3_s2')(pool4_helper)


inception_5a_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_5a/1x1',W_regularizer=l2(0.0002))(pool4_3x3_s2)

inception_5a_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_5a/3x3_reduce',W_regularizer=l2(0.0002))(pool4_3x3_s2)

inception_5a_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_5a/3x3',W_regularizer=l2(0.0002))(inception_5a_3x3_reduce)

inception_5a_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_5a/5x5_reduce',W_regularizer=l2(0.0002))(pool4_3x3_s2)

inception_5a_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_5a/5x5',W_regularizer=l2(0.0002))(inception_5a_5x5_reduce)

inception_5a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_5a/pool')(pool4_3x3_s2)

inception_5a_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_5a/pool_proj',W_regularizer=l2(0.0002))(inception_5a_pool)

inception_5a_output = merge([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj],mode='concat',concat_axis=1,name='inception_5a/output')


inception_5b_1x1 = Convolution2D(384,1,1,border_mode='same',activation='relu',name='inception_5b/1x1',W_regularizer=l2(0.0002))(inception_5a_output)

inception_5b_3x3_reduce = Convolution2D(192,1,1,border_mode='same',activation='relu',name='inception_5b/3x3_reduce',W_regularizer=l2(0.0002))(inception_5a_output)

inception_5b_3x3 = Convolution2D(384,3,3,border_mode='same',activation='relu',name='inception_5b/3x3',W_regularizer=l2(0.0002))(inception_5b_3x3_reduce)

inception_5b_5x5_reduce = Convolution2D(48,1,1,border_mode='same',activation='relu',name='inception_5b/5x5_reduce',W_regularizer=l2(0.0002))(inception_5a_output)

inception_5b_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_5b/5x5',W_regularizer=l2(0.0002))(inception_5b_5x5_reduce)

inception_5b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_5b/pool')(inception_5a_output)

inception_5b_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_5b/pool_proj',W_regularizer=l2(0.0002))(inception_5b_pool)

inception_5b_output = merge([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj],mode='concat',concat_axis=1,name='inception_5b/output')


pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='pool5/7x7_s2')(inception_5b_output)

loss3_flat = Flatten()(pool5_7x7_s1)

pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)

loss3_classifier = Dense(1000,name='loss3/classifier',W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)

loss3_classifier_act = Activation('softmax',name='prob')(loss3_classifier)


googlenet = Model(input=input, output=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act])
model = googlenet
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(trainX)

history = model.fit_generator(datagen.flow(trainX,trainY, batch_size=batch_size),
                              epochs = epochs, validation_data = (valX,valY),
                              verbose = 2, steps_per_epoch=trainX.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

# plt.show()

from sklearn.metrics import classification_report
y_true = valY
y_pred = model.predict(valX)
print(y_pred)
y_pred_list = list()
for index in range(y_pred.shape[0]):
	if y_pred[index][0] > y_pred[index][1]:
		y_pred_list.append(0)
	else:
		y_pred_list.append(1)

y_true_list = list()
for index in range(y_true.shape[0]):
	if y_true[index][0] > y_true[index][1]:
		y_true_list.append(0)
	else:
		y_true_list.append(1)

target_names = ['0', '1']
print(classification_report(y_true_list, y_pred_list, target_names=target_names))

model.save('my_model4.h5')
