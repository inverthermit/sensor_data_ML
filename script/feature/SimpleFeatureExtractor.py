import numpy as np
import matplotlib.pyplot as plt
import json
import copy
import os.path
import pandas as pd
from types import SimpleNamespace as Namespace
from feature.FeatureExtractor import FeatureExtractor
from util.TimeUtil import TimeUtil
class SimpleFeatureExtractor (FeatureExtractor):

    def getAccelerationFromFile(self,fileName):
        # print('Reading acceleration data from file: ',fileName)
        file_directory = fileName
        json_data=open(file_directory).read()
        # print(json_data[:100])
        entity = json.loads(json_data, object_hook=lambda d: Namespace(**d))
        # print('Lines of data: ',len(entity.data))
        # print(x.data[100].event.content)
        result = list()
        variable = 'acceleration' #quaternion acceleration
        for index in range(len(entity.data)):
        #     print(x.data[index].event.variable == 'acceleration')
            if(entity.data[index].event.variable == 'acceleration'):#quaternion
                result.append(entity.data[index].event.content)
        # print('File reading done. Total number of acceleration data: ',len(result))
        return result

    def getLabeledData(self, data, label):
        """For example: 1 2 3 12345 normal; 1 3 5 12346 tip; 1 4 8 12347 scallop"""
        print('Adding labels to data:' ,label)
        arr = copy.deepcopy(data)
        for ele in arr:
            ele.append(label)
        return arr

    def saveSimpleFeaturedData(self, path, dataFileNames, labels, resultFileName):
        if not os.path.isfile(path + resultFileName):
            allData = list()
            for i in range(len(dataFileNames)):
                fileArray = self.getAccelerationFromFile(path + dataFileNames[i])
                labeledData = self.getLabeledData(fileArray, labels[i])
                allData.extend(labeledData)
            np.savez(path + resultFileName, data=allData)
        else:
            print('Feature file:\'',path + resultFileName,'\' already exists.')

    def getSimpleFeaturedData(self, dataFilePath, label, returnDataFrame = True , startMin = None, endMin = None):
        allData = list()
        fileArray = self.getAccelerationFromFile(dataFilePath)
        labeledData = self.getLabeledData(fileArray, label)
        allData.extend(labeledData)
        data = np.array(allData)
        df = pd.DataFrame({'timeStamp': data[:,3], 'x': data[:,0], 'y': data[:,1], 'z': data[:,2], 'label': data[:,4]})
        df = df[['timeStamp','x', 'y', 'z', 'label']]

        if not (startMin is None and endMin is None):
            fileStartTimeStamp = df.iloc[0]['timeStamp']
            fileEndTimeStamp = df.iloc[len(df)-1]['timeStamp']

            startTimeStamp = fileStartTimeStamp
            if not startMin is None:
                startTimeStamp = fileStartTimeStamp + TimeUtil.getMillisecondFromMinute(
                    minute= startMin )

            if not endMin is None:
                endTimeStamp = fileStartTimeStamp + TimeUtil.getMillisecondFromMinute(
                    minute= endMin )
            df = df[(df['timeStamp'] >= startTimeStamp) & (df['timeStamp'] <= endTimeStamp)]
        df.sort_values('timeStamp')

        if returnDataFrame:
            return df

        return allData

    def getQuaternionFromFile(self,fileName):
        file_directory = fileName
        json_data=open(file_directory).read()
        entity = json.loads(json_data, object_hook=lambda d: Namespace(**d))
        result = list()
        variable = 'quaternion' #quaternion acceleration
        for index in range(len(entity.data)):
            if(entity.data[index].event.variable == 'quaternion'):#quaternion
                result.append(entity.data[index].event.content)
        return result

    def getQuaternionData(self, dataFilePath, label, returnDataFrame = True):
        allData = list()
        fileArray = self.getQuaternionFromFile(dataFilePath)
        labeledData = self.getLabeledData(fileArray, label)
        allData.extend(labeledData)

        if returnDataFrame:
            data = np.array(allData)
            df = pd.DataFrame({'w': data[:,0], 'x': data[:,1], 'y': data[:,2], 'z': data[:,3], 'timeStamp': data[:,4], 'label': data[:,len(data[0])-1]})
            df = df[['w','x','y','z','timeStamp','label']]
            return df

        return allData
