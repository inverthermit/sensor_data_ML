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

# extractor = TimeSeriesFeatureExtractor()
# extractor.getRollingMean([[1,2, 3, 4, 5, 6],[2,2, 3, 4, 5, 6]])
import pandas as pd


classificationNum = 3
path = '../../data/trials/'
tmpPath = '../../tmp/'
dataFileNames = ['4c.json']
labels = [0] #['normal', 'hole', 'scallop']
resultFileName = tmpPath + '4c.npz'

extractor = ProcessFeatureExtractor()
extractor.saveSimpleFeaturedData(path, dataFileNames, labels, resultFileName)

fileContent = np.load(path + resultFileName)
originData = fileContent['data']
extractor.tagProcessData('', originData)
