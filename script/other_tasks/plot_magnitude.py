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
from feature.TimeSeriesFeatureExtractor import TimeSeriesFeatureExtractor
from util.Util import Util
import pandas as pd


window_size = 8
timeSeriesExtractor = TimeSeriesFeatureExtractor()

path = '../../data/trials/'
dataFileNames = ['2b.json']
extractor = SimpleFeatureExtractor()


print('Magnitude:')
for ele in dataFileNames:
    df = timeSeriesExtractor.getSimpleFeaturedData(path + ele, 0)
    df = timeSeriesExtractor.insertRollingFeatures(df, window = window_size)
    df = df[df.index % window_size == 0]
    acceleration = df['Rolling_Mean_x'].as_matrix()
    print(acceleration)
    # acceleration = np.array(extractor.getAccelerationFromFile(path + ele))
    # acceleration = acceleration[:,0]
    # print(acceleration)
    # # y_plt = np.linalg.norm(acceleration,axis=1)
    y_plt = acceleration
    print(y_plt)
    x_plt = range(0,len(y_plt))
    plt.plot(x_plt, y_plt, 'r-')
    plt.show()
