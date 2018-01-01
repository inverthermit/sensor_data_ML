import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import json
import copy
# from types import SimpleNamespace as Namespace
from feature.FeatureExtractor import FeatureExtractor
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor

class TimeSeriesFeatureExtractor (SimpleFeatureExtractor):


    def insertRollingFeatures(self, data, window):
        """Set headers"""
        df = pd.DataFrame({'timeStamp': data[:,3], 'x': data[:,0], 'y': data[:,1], 'z': data[:,2], 'label': data[:,4]})
        df = df[['timeStamp','x', 'y', 'z', 'label']]
        #Calculate rolling mean and standard deviation using number of data set above
        rolling_mean_x = df['x'].rolling(window).mean()
        rolling_std_x = df['x'].rolling(window).std()
        rolling_mean_y = df['y'].rolling(window).mean()
        rolling_std_y = df['y'].rolling(window).std()
        rolling_mean_z = df['z'].rolling(window).mean()
        rolling_std_z = df['z'].rolling(window).std()
        df['Rolling_Mean_x'] = rolling_mean_x
        df['Rolling_Std_x'] = rolling_std_x
        df['Rolling_Mean_y'] = rolling_mean_y
        df['Rolling_Std_y'] = rolling_std_y
        df['Rolling_Mean_z'] = rolling_mean_z
        df['Rolling_Std_z'] = rolling_std_z

        df = df[['timeStamp','x', 'y', 'z', 'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z','Rolling_Std_x','Rolling_Std_y','Rolling_Std_z', 'label']]
        train_data = df.values
        train_no_nan = []
        for i in range(0, len(train_data)):
            if(not np.isnan(train_data[i]).any()):
                train_no_nan.append(train_data[i])
        print('origin:',len(df))
        print('trimmed:', len(train_no_nan))
        return np.array(train_no_nan)
