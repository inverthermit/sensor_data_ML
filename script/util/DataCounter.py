import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import sys
sys.path.append('../')
from os import listdir
from os.path import isfile, join
from types import SimpleNamespace as Namespace
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor

path = '../../data/trials/'
fileNames = [f for f in listdir(path) if isfile(join(path, f))]


extractor = SimpleFeatureExtractor()
for fileName in fileNames:
    data = extractor.getAccelerationFromFile(path+fileName)
    print(len(data),',',fileName)
