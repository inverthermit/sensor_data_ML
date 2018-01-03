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

path = '../../data/'
dataFileNames = ['drain.json','Pin hole tip.json','Scallop tip.json']
extractor = SimpleFeatureExtractor()

for ele in dataFileNames:
    acceleration = np.array(extractor.getAccelerationFromFile(path + ele))
    x = acceleration[:,0]
    y = acceleration[:,1]
    z = acceleration[:,2]
    print('File: ',path+ele)
    print('Mean(x,y,z):', np.mean(x), np.mean(y), np.mean(z))
    print('Variance(x,y,z):', np.var(x), np.var(y), np.var(z))
