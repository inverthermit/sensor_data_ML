import copy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os.path
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace
# from SimpleNamespace import SimpleNamespace as Namespace
from util.Util import Util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../')
rootDir = '../../'
path = rootDir + Util.getConfig('labeled_csv')
n_clusters = Util.getConfig('number_of_clusters')
data = pd.read_csv(path + 'kcm_n_cluster'+str(n_clusters)+'.csv')
print(data)
data['sum'] =data['cluster'+str(0)]
for i in range(1, n_clusters):
    data['sum'] += data['cluster'+str(i)]
# print(data)
train_data_title = []
for i in range(n_clusters):
    data['percentage'+str(i)] = data['cluster'+str(i)]*100/data['sum']
    train_data_title.append('percentage'+str(i))
train_data_title.append('label')
print(data[train_data_title])
train_data = data[train_data_title].as_matrix()

# print(len(train_data))
# print(train_data)
np.random.shuffle(train_data)
# print(len(train_data))
# print(train_data)

from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# import graphviz
# clf = DecisionTreeClassifier(random_state=0)
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=3)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(max_iter = 1000)
X = train_data[:,:len(train_data[0])-1]
y = train_data[:,len(train_data[0])-1]
# print(X,y)
cross_val_score = cross_val_score(clf, X, y, cv=10)
print(cross_val_score)
print(np.mean(cross_val_score))

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_pred = cross_val_predict(clf,X,y,cv=10)
conf_mat = confusion_matrix(y,y_pred)
print(conf_mat)
#
from sklearn.metrics import classification_report
target_names = ['0', '1']
print(classification_report(y,y_pred, target_names=target_names))
# print(cross_val_score(clf, X, y, cv=10))
# clf.fit(X,y)
# dot_data = tree.export_graphviz(clf, out_file='D:/graphviz-2.38/release/bin/data/decision_tree.dot')
