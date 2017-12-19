import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from types import SimpleNamespace as Namespace
from feature.FeatureExtractor import FeatureExtractor
class SimpleFeatureExtractor (FeatureExtractor):

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
