import copy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os.path
from sklearn.externals import joblib
import sys
sys.path.append('../')
from SimpleNamespace import SimpleNamespace as Namespace
from feature.FeatureTransformation import FeatureTransformation
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor
from feature.TimeSeriesFeatureExtractor import TimeSeriesFeatureExtractor
from util.Util import Util
import pandas as pd

input_one = input('Please input two file names, separated as space, used as training data. For example: 1a.json 1b.json\n')
input_two = input('Please input one file names, together with the label, separated with space. This is used as testing data. For example: 1a.json 0\n')


sys.path.append('../')
classificationNum = 3
rootDir = '../../'
path = rootDir + Util.getConfig('trials_folder_path')
tmpPath = rootDir + Util.getConfig('tmp_path')

extractor = TimeSeriesFeatureExtractor()
featureTransformer = FeatureTransformation()

train_file_names = None
test_file_names = None
labels = None

train_file_names = input_one.split(' ')
test_file_names = input_two.split(' ')[0]
train_labels = [0,1]
test_labels = int(input_two.split(' ')[1])


train_data = list()
test_data = list()
for index, label in enumerate(train_labels):
    train_df = extractor.getSimpleFeaturedData(path + train_file_names[index], label)
    train_data.append(train_df)
test_df = extractor.getSimpleFeaturedData(path + test_file_names, test_labels)
test_data.append(test_df)

print('Scaling the y&z axises...')
for index in range(len(train_data)):
    train_data[index] = featureTransformer.scaleYZAxis(train_data[index])
for index in range(len(test_data)):
    test_data[index] = featureTransformer.scaleYZAxis(test_data[index])

x_train = None
y_train = None
x_test = None
y_test = None

dfAll = None
for df in train_data:
    df = extractor.insertRollingFeatures(df, window = 350)
    if dfAll is None:
        dfAll = df
        continue
    else:
        dfAll = dfAll.append(df)
train_data = dfAll
dfAll = None
for df in test_data:
    df = extractor.insertRollingFeatures(df, window = 350)
    if dfAll is None:
        dfAll = df
        continue
    else:
        dfAll = dfAll.append(df)
test_data = dfAll

train_data = train_data[['x', 'y', 'z',
    'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z',
    'Rolling_Std_x','Rolling_Std_y','Rolling_Std_z',
    'label']]
test_data = test_data[['x', 'y', 'z',
    'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z',
    'Rolling_Std_x','Rolling_Std_y','Rolling_Std_z',
    'label']]

train_data = train_data.as_matrix()
test_data = test_data.as_matrix()

random.shuffle(train_data)
random.shuffle(test_data)

x_train = train_data[:,:-1]
y_train = train_data[:,-1]
x_test = test_data[:,:-1]
y_test = test_data[:,-1]

numTree = 9
"""Random Forest"""
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=numTree, n_jobs = 8)
model = rf_model
print('Random Forest(',numTree,'):')

model.fit(x_train,y_train)
print('Training score: ',model.score(x_train,y_train))
print('Testing score: ', model.score(x_test,y_test))

from sklearn.metrics import classification_report
y_true = y_test
y_pred = model.predict(x_test)
print(labels)
print(classification_report(y_true, y_pred, target_names = ['0','1']))
k=input("press close to exit") 
