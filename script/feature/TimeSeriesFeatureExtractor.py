import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import json
import copy
from types import SimpleNamespace as Namespace
from feature.FeatureExtractor import FeatureExtractor
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor

class TimeSeriesFeatureExtractor (SimpleFeatureExtractor):


    def getRollingMean(self, data):
        print('getRollingMean')

        df = pd.DataFrame({'timeStamp': data[:,4], 'x': data[:,4],
        'y': data[:,4], 'z': data[:,4], 'w': data[:,4],
        'mean':data[:,4], 'variance': data[:,4], 'label': data[:,4]})
        print(df)


    def getAccelerationFromFile(self,fileName):
        print('Reading acceleration data from file: ',fileName)
        file_directory = fileName
        json_data=open(file_directory).read()
        # print(json_data[:100])
        entity = json.loads(json_data, object_hook=lambda d: Namespace(**d))
        print('Lines of data: ',len(entity.data))
        # print(x.data[100].event.content)
        result = list()
        variable = 'acceleration' #quaternion acceleration
        for index in range(len(entity.data)):
        #     print(x.data[index].event.variable == 'acceleration')
            if(entity.data[index].event.variable == 'acceleration'):#quaternion
                result.append(entity.data[index].event.content)
        print('File reading done. Total number of acceleration data: ',len(result))
        return result

    def getLabeledData(self, data, label):
        """For example: 1 2 3 12345 normal; 1 3 5 12346 tip; 1 4 8 12347 scallop"""
        print('Adding labels to data:' ,label)
        arr = copy.deepcopy(data)
        for ele in arr:
            ele.append(label)
        return arr
