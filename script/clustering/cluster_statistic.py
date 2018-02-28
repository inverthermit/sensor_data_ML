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
import pandas as pd

from util.Util import Util

sys.path.append('../')
rootDir = '../../'
n_clusters = Util.getConfig('number_of_clusters')
path = rootDir + Util.getConfig('trials_folder_path')
tmpPath = rootDir + Util.getConfig('image_save_folder')
savePath = rootDir + Util.getConfig('pic_to_np_array')
npzPath = rootDir + Util.getConfig('image_filename_belong_cluster_npz')
csvPath = rootDir + Util.getConfig('clustering_result_csv_folder')
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    print(kmeans.labels_)
    # print(kmeans.predict([[0, 0], [4, 4]]))
    print(kmeans.cluster_centers_)

    endTime = datetime.now()
    print('Running Time:',str(endTime - startTime),' ',str(startTime), str(endTime))


    picFileNames = np.array([f for f in listdir(tmpPath) if isfile(join(tmpPath, f))])
    result = np.transpose(np.vstack((picFileNames,kmeans.labels_)))
    np.savez(npzPath, data = result)


genClusterResult()
print('genclusterresult done.')
filename_cluster = np.load(npzPath)['data']
print(filename_cluster)

raw_data_filenames = np.array([])
for name in filename_cluster[:,0]:
    raw_data_filenames = np.append(raw_data_filenames, [name.split('.json')[0]])

result = np.transpose(np.vstack((raw_data_filenames, filename_cluster[:,1])))

"""Do the counting"""
unique_files = np.unique(result[:,0])
unique_clusters = [str(i) for i in range(n_clusters)]
print('unique clusters;',unique_clusters)
print(unique_files)

data = {}
for key in unique_files:
    key_row = [row[1] for row in result if row[0]==key]
    unique, counts = np.unique(key_row, return_counts=True)
    counts_sum = sum(counts)
    value = dict(zip(unique, counts))#np.round_(counts/counts_sum*1.0, 2)
    data[key] = value

count_arr = []

for key in data:
    row = key
    arr_row = [key]
    file_content = data[key]
    for cluster in unique_clusters:
        if cluster not in file_content:
            row += ' 0'
            arr_row.append('0')
            continue
        row += ' '+str(file_content[cluster])
        arr_row.append(str(file_content[cluster]))
    count_arr.append(arr_row)

df_sum_title = ['sum']
# unique_clusters = [str(i) for i in range(5)]
df_start_title = ['trail' ]
df_cluster_title = [('cluster'+ ele) for ele in unique_clusters]

df_title = df_start_title
df_title.extend(df_cluster_title)

df = pd.DataFrame(count_arr,columns=df_title)
print(df)
df.to_csv(csvPath+'kcm_n_cluster'+str(n_clusters)+'.csv', sep=',', encoding='utf-8',index=False)

# df_title
#
# df_percentage_title = [('percent'+ ele) for ele in unique_clusters]


# if '.json' in trailName:
#     trailName = trailName.replace('.json','')
# rootDir = '../../'
# with open(rootDir + Util.getConfig('trials_info_path')) as json_data_file:
#     data = json.load(json_data_file)
# trialInfoObj = None
# for ele in data:
#     if ele['Trial'] == trailName:
#         trialInfoObj = ele
#         break

# df['purchase'].astype(str).astype(int)

















"""End of file"""
