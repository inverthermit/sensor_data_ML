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

# extractor = TimeSeriesFeatureExtractor()
# extractor.getRollingMean([[1,2, 3, 4, 5, 6],[2,2, 3, 4, 5, 6]])
import pandas as pd

classificationNum = 3
path = '../../data/'
tmpPath = '../tmp/'
dataFileNames = ['drain.json','Pin hole tip.json','Scallop tip.json']
labels = [0, 1, 1] #['normal', 'hole', 'scallop']
resultFileName = tmpPath + 'simpleFeatures012.npz'

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
print(rand_data[0])
X_rand = rand_data[:,1:10]
y_rand = rand_data[:,10]
# print('888888888888', X_rand, '---------', y_rand)

heldout_len = int(len(X_rand)*0.8)
x_train = X_rand[:heldout_len]
y_train = y_rand[:heldout_len]
x_test = X_rand[heldout_len:]
y_test = y_rand[heldout_len:]
# X = data[:,:3]
# y = data[:,4]

for numTree in range(9,11):
    if(numTree%2 == 0):
        continue
    """Random Forest"""
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=numTree)
    model = rf_model
    print('Random Forest(',numTree,'):')

    # """Artificial Neural Network"""
    # from sklearn.neural_network import MLPClassifier
    # ann_model = MLPClassifier()
    # model = ann_model
    # print('ANN:')
    #
    # """SVM"""
    # from sklearn.svm import SVC
    # svm_model = SVC()
    # model = svm_model
    # print('SVM:')

    model.fit(x_train,y_train)
    print('Training score: ',model.score(x_train,y_train))
    print('Testing score: ', model.score(x_test,y_test))

    from sklearn.metrics import classification_report
    y_true = y_test
    y_pred = model.predict(x_test)
    target_names = ['0', '1', '2']
    print(classification_report(y_true, y_pred, target_names=target_names))























print('')
