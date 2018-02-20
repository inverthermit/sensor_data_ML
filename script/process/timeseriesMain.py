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
from util.Util import Util
# extractor = TimeSeriesFeatureExtractor()
# extractor.getRollingMean([[1,2, 3, 4, 5, 6],[2,2, 3, 4, 5, 6]])
import pandas as pd

classificationNum = 3

window_size = 350
rootDir = '../../'
path = rootDir + Util.getConfig('trials_folder_path')
tmpPath = rootDir + Util.getConfig('tmp_path')
dataFileNames = ['4a.json','4e.json','2a.json']
dataFileMinutes = np.array([[None,None],[None,None],[None,None]])
labels = [0,1,1] #['normal', 'hole', 'scallop']

testFileNames = ['0a.json']
testFileMinutes = np.array([[0,4]])
testFileLabels = [0]


min_samples_leaf = 0.04
is_heldout = True
x_columns = [
                'x','Rolling_Mean_x','Rolling_Std_x','Max_x','Min_x',
                'y', 'z',
                'Rolling_Mean_y','Rolling_Mean_z',
                'Rolling_Std_y','Rolling_Std_z'
                ]
y_columns = ['label']



featureTransformer = FeatureTransformation()
timeSeriesExtractor = TimeSeriesFeatureExtractor()
dfAll = None
for index, dataFileName in enumerate(dataFileNames):
    print(dataFileName)
    df = timeSeriesExtractor.getSimpleFeaturedData(path + dataFileName, labels[index]
    ,startMin = dataFileMinutes[index,0],endMin = dataFileMinutes[index,1])
    # print(len(df))
    # print(df)
    df = featureTransformer.scaleYZAxis(df)
    # print(df)
    # print(len(df))
    df = timeSeriesExtractor.insertRollingFeatures(df, window = window_size)
    # print(len(df))
    print(path + dataFileName, labels[index],len(df))
    if dfAll is None:
        dfAll = df
        continue
    else:
        dfAll = dfAll.append(df)
# dfAll.to_csv('./dataAll.csv', sep=',')
print(len(dfAll))


"""Test data"""
test_dfAll = None
for index, dataFileName in enumerate(testFileNames):
    df = timeSeriesExtractor.getSimpleFeaturedData(path + dataFileName, testFileLabels[index]
    ,startMin = testFileMinutes[index,0],endMin = testFileMinutes[index,1])
    # df = featureTransformer.scaleYZAxis(df)
    df = timeSeriesExtractor.insertRollingFeatures(df, window = window_size)
    # print(len(df))
    print(path + dataFileName, testFileLabels[index],len(df))
    if test_dfAll is None:
        test_dfAll = df
        continue
    else:
        test_dfAll = test_dfAll.append(df)

print(len(test_dfAll))



dfAll = dfAll[x_columns + y_columns]
data = dfAll.as_matrix()

print(dfAll)
plotData = dfAll[['Rolling_Std_x', 'Rolling_Std_y', 'Rolling_Std_z','label']].as_matrix()
# 'x','y','z'
# 'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z'
# 'Rolling_Std_x', 'Rolling_Std_y', 'Rolling_Std_z'
class0 = np.array([row for row in plotData if row[len(row)-1]==0])
class1 = np.array([row for row in plotData if row[len(row)-1]==1])
class2 = np.array([row for row in plotData if row[len(row)-1]==2])
# print(class0)
# print(class1)
# plt.plot(class0[:,1],class0[:,2],'r.')
# plt.plot(class1[:,1],class1[:,2],'b.')
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.patches as mpatches
#
#
# fig = plt.figure()
# red_patch = mpatches.Patch(color='red', label='The red data')
# plt.legend(handles=[red_patch])
# ax = Axes3D(fig)
# start = 3000
# end = 3800
# # print(class0)
# ax.scatter(class0[start:end,0], class0[start:end,1], class0[start:end,2],'b.')
# ax.scatter(class1[start:end,0], class1[start:end,1], class1[start:end,2],'r.')
# ax.scatter(class2[start:end,0], class2[start:end,1], class2[start:end,2],'g.')
# plt.show()



test_dfAll = test_dfAll[x_columns + y_columns]
test_data = test_dfAll.as_matrix()


"""['timeStamp',
    'x', 'y', 'z',
    'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z',
    'Rolling_Std_x','Rolling_Std_y','Rolling_Std_z',
    'label']
"""

print('****************Start to run classifications***************')
rand_data = np.array(copy.deepcopy(data))
np.random.shuffle(rand_data)
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
    rf_model = RandomForestClassifier(n_estimators=numTree, min_samples_leaf  = min_samples_leaf, n_jobs =8)
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

    # """knn"""
    # from sklearn.neighbors import KNeighborsClassifier
    # model = KNeighborsClassifier(n_neighbors=numTree)
    #
    # """SGD"""
    # from sklearn import linear_model
    # model = linear_model.SGDClassifier(max_iter=1000,tol = 1000)

    # """adaboost"""
    # from sklearn.model_selection import cross_val_score
    # from sklearn.datasets import load_iris
    # from sklearn.ensemble import AdaBoostClassifier

    model.fit(x_train,y_train)

    # from sklearn.tree import export_graphviz
    # export_graphviz(model.estimators_[0],
    #                 feature_names=x_columns,
    #                 filled=True,
    #                 rounded=True)

    # os.system('dot -Tpng tree.dot -o tree.png')


    print('Training score: ',model.score(x_train,y_train))
    print('Testing score: ', model.score(x_test,y_test))

    from sklearn.metrics import classification_report
    y_true = y_test
    y_pred = model.predict(x_test)
    target_names = ['0','1']
    print(classification_report(y_true, y_pred, target_names=target_names))

    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_true,y_pred, labels = [0,1])
    print(conf_mat)

    if is_heldout:

        y_pred = cross_val_predict(model,x_train,y_train,cv=10)
        conf_mat = confusion_matrix(y_train,y_pred)
        print(conf_mat)























print('')
