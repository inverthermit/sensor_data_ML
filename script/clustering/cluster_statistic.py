from __future__ import division
import os
import numpy as np
from datetime import datetime
from os.path import isfile, join
from os import listdir
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace
# from SimpleNamespace import SimpleNamespace as Namespace
import random
import os.path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

from util.Util import Util

sys.path.append('../')
classificationNum = 3
rootDir = '../../'
path = rootDir + Util.getConfig('trials_folder_path')
tmpPath = rootDir + Util.getConfig('tmp_path') +'/all/'#big
savePath = rootDir + Util.getConfig('pic_to_np_array') +'/all/'#big
width = 80
height = 60


def getAllData():
    fileNames = [f for f in listdir(savePath) if isfile(join(savePath, f))]
    data_total = None
    num_pic_in_file = list()
    for file_name in fileNames:
        file_data = np.load(savePath + file_name)

        if data_total is None:
            data_total = file_data['data']
        else:
            data_total = np.vstack((data_total, file_data['data']))
        num_pic_in_file.append([file_name, len(file_data['data'])])
    return data_total, np.array(num_pic_in_file)

def genClusterResult():
    X, num_pic_in_file = getAllData()#[:2000]
    print((num_pic_in_file[:,1].astype(int)))

    startTime = datetime.now()
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    print(kmeans.labels_)
    # print(kmeans.predict([[0, 0], [4, 4]]))
    print(kmeans.cluster_centers_)

    endTime = datetime.now()
    print('Running Time:',str(endTime - startTime),' ',str(startTime), str(endTime))


    picFileNames = np.array([f for f in listdir(tmpPath) if isfile(join(tmpPath, f))])
    result = np.transpose(np.vstack((picFileNames,kmeans.labels_)))
    np.savez(savePath + 'kmeans7kAll.npz', data = result)


genClusterResult()
print('genclusterresult done.')
filename_cluster = np.load(savePath + 'kmeans7kAll.npz')['data']
# print(filename_cluster)

raw_data_filenames = np.array([])
for name in filename_cluster[:,0]:
    raw_data_filenames = np.append(raw_data_filenames, [name[:2]])

result = np.transpose(np.vstack((raw_data_filenames, filename_cluster[:,1])))

"""Do the counting"""
unique_files = np.unique(result[:,0])
unique_clusters = np.unique(result[:,1])
print(unique_files)

data = {}
for key in unique_files:
    key_row = [row[1] for row in result if row[0]==key]
    unique, counts = np.unique(key_row, return_counts=True)
    counts_sum = sum(counts)
    # dict(zip(unique, counts))
    value = dict(zip(unique, counts))#np.round_(counts/counts_sum*1.0, 2)
    data[key] = value
# print(json.dumps(data))
# print(data)

for key in data:
    # print(key+':')
    row = key
    file_content = data[key]
    clusters = ['0','1','2','3','4']
    for cluster in clusters:
        if cluster not in file_content:
            row += ' 0'
            continue
        row += ' '+str(file_content[cluster])
    print(row)



# unique, counts = numpy.unique(a, return_counts=True)
# dict(zip(unique, counts))

# print(result)

# a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
# >>> unique, counts = numpy.unique(a, return_counts=True)
# >>> dict(zip(unique, counts))















"""End of file"""
