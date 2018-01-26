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
import pandas as pd

# Test how our algorithm performs in others' data.
rootDir = '../../data/AReM/'
fileName1 = '/dataset'
fileName2 = '.csv'
folders = ['bending1','bending2','cycling','lying','sitting','standing','walking']
is_heldout = True
min_samples_leaf = 1
# print(df)
dfAll = None
for index,folder in enumerate(folders):
    for i in range(1,14):
        filePath = rootDir + folder + fileName1 + str(i) + fileName2
        import os.path
        if not os.path.isfile(filePath):
            continue
        df = pd.read_csv(filePath, comment='#'
        ,names = ['time','avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23'])
        df['label'] = folder
        print(filePath, folder, len(df))
        if dfAll is None:
            dfAll = df
            continue
        else:
            dfAll = dfAll.append(df)


"""Test data"""
test_dfAll = None
for index,folder in enumerate(folders):
    for i in range(14,16):
        filePath = rootDir + folder + fileName1 + str(i) + fileName2
        import os.path
        if not os.path.isfile(filePath):
            continue
        df = pd.read_csv(filePath, comment='#'
        ,names = ['time','avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23'])
        df['label'] = folder
        print(filePath, folder, len(df))
        if test_dfAll is None:
            test_dfAll = df
            continue
        else:
            test_dfAll = test_dfAll.append(df)

print(len(test_dfAll))


dfAll = dfAll.dropna(subset=['avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23','label'])
dfAll = dfAll[['avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23','label']]
data = dfAll.as_matrix()

test_dfAll = test_dfAll.dropna(subset=['avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23','label'])
test_dfAll = test_dfAll[['avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23','label']]
test_data = test_dfAll.as_matrix()
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
    target_names = folders
    print(classification_report(y_true, y_pred, target_names=target_names))

    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_true,y_pred, labels = folders)
    print(conf_mat)

    if is_heldout:

        y_pred = cross_val_predict(model,x_train,y_train,cv=10)
        conf_mat = confusion_matrix(y_train,y_pred)
        print(conf_mat)
