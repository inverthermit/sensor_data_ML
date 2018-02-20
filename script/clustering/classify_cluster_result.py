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
path = rootDir + 'data/'

data = pd.read_csv(path + 'clustering_features_all.csv')
print(data)

data['sum'] = data['cluster0']+data['cluster1']+data['cluster2']+data['cluster3']+data['cluster4']
data['percentage0'] = data['cluster0']*100/data['sum']
data['percentage1'] = data['cluster1']*100/data['sum']
data['percentage2'] = data['cluster2']*100/data['sum']
data['percentage3'] = data['cluster3']*100/data['sum']
data['percentage4'] = data['cluster4']*100/data['sum']

# print(data)

train_data = data[['percentage0','percentage1','percentage2','percentage3','percentage4','label']].as_matrix()
# print(len(train_data))
# print(train_data)
np.random.shuffle(train_data)
# print(len(train_data))
print(train_data)

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
clf = DecisionTreeClassifier(random_state=0)
X = train_data[:,:len(train_data[0])-1]
y = train_data[:,len(train_data[0])-1]
# print(X,y)
cross_val_score = cross_val_score(clf, X, y, cv=10)
print(cross_val_score)
print(np.mean(cross_val_score))
# print(cross_val_score(clf, X, y, cv=10))
clf.fit(X,y)
dot_data = tree.export_graphviz(clf, out_file='D:/graphviz-2.38/release/bin/data/decision_tree.dot')
