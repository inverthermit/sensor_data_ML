import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import json
import copy
from types import SimpleNamespace as Namespace
from feature.FeatureExtractor import FeatureExtractor
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor

class TimeSeriesFeatureExtractor (SimpleFeatureExtractor):


    def insertRollingFeatures(self, df, window):
        #[['timeStamp','x', 'y', 'z', 'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z','Rolling_Std_x','Rolling_Std_y','Rolling_Std_z', 'label']]

        #Calculate rolling mean and standard deviation using number of data set above
        df['Rolling_Mean_x'] = df['x'].rolling(window).mean()
        df['Rolling_Std_x'] = df['x'].rolling(window).std()
        df['Rolling_Mean_y'] = df['y'].rolling(window).mean()
        df['Rolling_Std_y'] = df['y'].rolling(window).std()
        df['Rolling_Mean_z'] = df['z'].rolling(window).mean()
        df['Rolling_Std_z'] = df['z'].rolling(window).std()
        df = df.dropna(subset=['Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z'])
        return df
