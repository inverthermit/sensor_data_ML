import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace
from feature.FeatureTransformation import FeatureTransformation
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor
from feature.TimeSeriesFeatureExtractor import TimeSeriesFeatureExtractor
from feature.ProcessFeatureExtractor import ProcessFeatureExtractor
from feature.QuanternionFeatureExtractor import QuanternionFeatureExtractor
from util.Util import Util


import pandas as pd

classificationNum = 3
rootDir = '../../'
path = rootDir + Util.getConfig('trials_folder_path')
tmpPath = rootDir + Util.getConfig('tmp_path')

dataFileNames = [
'1a.json',
'1b.json',
'3b.json',
'4e.json',
'4f.json',
'3g.json',
'2b.json',
'3c.json',
'4c.json',
'4d.json',
'3d.json',
'4b.json',
'4g.json'
]
labels = [0,0,0,0,0,0,1,1,1,1,1,1,1]

# dataFileNames = ['0a.json','4a.json','4b.json','4c.json', '4d.json','4e.json','4f.json','4g.json','4h.json','4i.json']
# labels = [0,0, 1, 1,1,1,1,1,0,0] #['normal', 'hole', 'scallop']

# dataFileNames = [
# '0a.json','1a.json','1b.json','1c.json','1d.json',
# '1e.json','2a.json','3a.json','4a.json','4h.json',
# '4i.json','5a.json','5b.json','5c.json','5d.json',
# '5e.json','5f.json','5g.json','5h.json',
#
# '3b.json','4e.json','4f.json','0b.json','3f.json',
# '3e.json','0c.json','3d.json','4b.json','4g.json',
# '2b.json','3c.json','4c.json','4d.json','3g.json'
#
# ]
#
# labels = [
# 0,0,0,0,0,
# 0,0,0,0,0,
# 0,0,0,0,0,
# 0,0,0,0,
# 1,1,1,1,1,
# 1,1,1,1,1,
# 1,1,1,1,1,1
# ]

"""no 0b.json"""
# dataFileNames = [
# '0a.json','1a.json','1b.json','1c.json','1d.json',
# '1e.json','2a.json','3a.json','4a.json','4h.json',
# '4i.json','5a.json','5b.json','5c.json','5d.json',
# '5e.json','5f.json','5g.json','5h.json',
#
# '3b.json','4e.json','4f.json','3f.json',
# '3e.json','0c.json','3d.json','4b.json','4g.json',
# '2b.json','3c.json','4c.json','4d.json','3g.json'
#
# ]
#
# labels = [
# 0,0,0,0,0,
# 0,0,0,0,0,
# 0,0,0,0,0,
# 0,0,0,0,
# 1,1,1,1,
# 1,1,1,1,1,
# 1,1,1,1,1,1
# ]

"""All A mount_type"""
# dataFileNames = ['0a.json','1a.json','1b.json','2a.json','4a.json','4h.json',
# '4e.json','4f.json','0b.json','0c.json','4b.json','4g.json','2b.json','4c.json','4d.json']
# labels = [0,0,0,0,0,0,
# 1,1,1,1,1,1,1,1,1]

# dataFileNames = ['0a.json','1a.json','1b.json','2a.json','4a.json',#'4h.json',
# '4e.json','4f.json','0b.json','0c.json','4b.json','4g.json','2b.json','4c.json',#'4d.json'
# ]
# labels = [0,0,0,0,0,0,
# 1,1,1,1,1,1,1,1,1]

testFileNames = ['1c.json','1d.json']
testFileLabels = [0,0]

is_heldout = False

processExtractor = ProcessFeatureExtractor()
dfAll = None
for index, dataFileName in enumerate(dataFileNames):
    df = processExtractor.getQuaternionData(path + dataFileName, labels[index])
    df = processExtractor.insertProcessData(df,dataFileName)
    # print(len(df))
    print(path + dataFileName, labels[index],len(df))
    if dfAll is None:
        dfAll = df
        continue
    else:
        dfAll = dfAll.append(df)

print(len(dfAll))
# print(dfAll)
"""Test data"""
test_dfAll = None
for index, dataFileName in enumerate(testFileNames):
    df = processExtractor.getQuaternionData(path + dataFileName, testFileLabels[index])
    df = processExtractor.insertProcessData(df,dataFileName)
    # print(len(df))
    print(path + dataFileName, testFileLabels[index],len(df))
    if test_dfAll is None:
        test_dfAll = df
        continue
    else:
        test_dfAll = test_dfAll.append(df)

print(len(test_dfAll))

dfAll = dfAll[['w', 'x', 'y', 'z','label']]
# data = dfAll.as_matrix()
data = QuanternionFeatureExtractor.squeezeWXYZ(dfAll)[0::20]

test_dfAll = test_dfAll[['w', 'x', 'y', 'z','label']]
test_data = QuanternionFeatureExtractor.squeezeWXYZ(test_dfAll)[0::20]

print(len(data),len(test_data))
# print(QuanternionFeatureExtractor.squeezeWXYZ(df))



"""
    'x', 'y', 'z',
    'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z',
    'Rolling_Std_x','Rolling_Std_y','Rolling_Std_z',
    'ngimu_mount_type','ngimu_position', 'bath_start_depth',
    'immersion', 'lance_flow',
    'label'
"""

print('****************Start to run classifications***************')
rand_data = np.array(copy.deepcopy(data))
random.shuffle(rand_data)
print(rand_data[0])
X_rand = rand_data[:,:len(data[0])-1]
y_rand = rand_data[:,len(data[0])-1]
# print('888888888888', X_rand, '---------', y_rand)

heldout_len = int(len(X_rand)*0.8)
x_train = X_rand[:heldout_len]
y_train = y_rand[:heldout_len]
x_test = []
y_test = []
if is_heldout:
    x_test = X_rand[heldout_len:]
    y_test = y_rand[heldout_len:]
else:
    x_test = test_data[:,:len(test_data[0])-1]
    y_test = test_data[:,len(test_data[0])-1]

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


    model.fit(x_train,y_train)
    print('Training score: ',model.score(x_train,y_train))
    print('Testing score: ', model.score(x_test,y_test))

    from sklearn.externals import joblib
    joblib.dump(model, tmpPath+'rf80features.pkl')

    from sklearn.metrics import classification_report
    y_true = y_test
    y_pred = model.predict(x_test)
    target_names = ['0', '1', '2','3']
    print(classification_report(y_true, y_pred, target_names=target_names))























print('')
