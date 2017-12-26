import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor

classificationNum = 3
path = '../../data/'
dataFileNames = ['drain.json','Pin hole tip.json','Scallop tip.json']
labels = [0, 1, 1] #['normal', 'hole', 'scallop']
resultFileName = 'simpleFeatures01.npz'



if not os.path.isfile(path + resultFileName):
    print('****************Start to extract features***************')
    extractor = SimpleFeatureExtractor()
    allData = list()
    for i in range(len(dataFileNames)):
        fileArray = extractor.getAccelerationFromFile(path + dataFileNames[i])
        print(fileArray[100])

        labeledData = extractor.getLabeledData(fileArray, labels[i])
        print(labeledData[100])
        allData.extend(labeledData)
    print('Length of the whole dataset: ',len(allData))


    np.savez(path + resultFileName, data=allData)


print('****************Start to run classifications***************')
fileContent = np.load(path + resultFileName)
data = fileContent['data']
print(len(data))
rand_data = copy.deepcopy(data)
random.shuffle(rand_data)
# extract a stack of 28x28 bitmaps
X_rand = rand_data[:,:3]
y_rand = rand_data[:,4]

# X_rand = digits[:, 0:784]
# y_rand = digits[:, 784:785]
heldout_len = int(len(X_rand)*0.8)
x_train = X_rand[:heldout_len]
y_train = y_rand[:heldout_len]
x_test = X_rand[heldout_len:]
y_test = y_rand[heldout_len:]
# X = data[:,:3]
# y = data[:,4]

for numTree in range(1,11):
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
    target_names = ['0', '1']
    print(classification_report(y_true, y_pred, target_names=target_names))
