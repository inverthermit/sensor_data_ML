import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace
from feature.ProcessFeatureExtractor import ProcessFeatureExtractor
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor
from feature.TimeSeriesFeatureExtractor import TimeSeriesFeatureExtractor
from feature.FeatureTransformation import FeatureTransformation
from util.Util import Util
# extractor = TimeSeriesFeatureExtractor()
# extractor.getRollingMean([[1,2, 3, 4, 5, 6],[2,2, 3, 4, 5, 6]])
import pandas as pd

print(Util.getConfig('trials_info_path'))

classificationNum = 3
rootDir = '../../'
path = rootDir + Util.getConfig('trials_folder_path')
tmpPath = rootDir + Util.getConfig('tmp_path')
dataFileNames = ['4a.json','4b.json','4c.json']
labels = [0, 1, 2] #['normal', 'hole', 'scallop']

processExtractor = ProcessFeatureExtractor()
timeseriesExtractor = TimeSeriesFeatureExtractor()
for index, val in dataFileNames:
    #Read from json file ['timeStamp,x,y,z,label']
    df = processExtractor.getSimpleFeaturedData(path+dataFileNames[0], labels[0], True)

    #Add process features
    processData = extractor.tagProcessData(df, val)

    #Feature Scaling
    data = featureTransformer.scaleYZAxis(data)

    # Rotation
    data = featureTransformer.rotateYZAxis(data)

    # Add rolling window features
    train_no_nan = extractor.insertRollingFeatures(data, window = 350)
    rand_data = np.array(copy.deepcopy(train_no_nan))

    # Get Training data

extractor = ProcessFeatureExtractor()
df = extractor.getSimpleFeaturedData(path+dataFileNames[0], labels[0], True)
print(extractor.tagProcessData(df, '4c', True))
