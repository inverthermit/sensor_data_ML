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

input_shape = (img_width, img_height, 1)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu', input_shape = input_shape))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation = "softmax"))


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

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
