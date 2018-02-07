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
    test_matrix = df[['x','y','z']]
