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

path = '../../data/trials/'
dataFileNames = ['1a.json','1c.json','1e.json']
extractor = SimpleFeatureExtractor()


print('Mean')
for ele in dataFileNames:
    acceleration = np.array(extractor.getAccelerationFromFile(path + ele))
    x = acceleration[:,0]
    y = acceleration[:,1]
    z = acceleration[:,2]
    # print('File: ',path+ele)
    print(np.mean(x), np.mean(y), np.mean(z)) #'Mean(x,y,z):',

print('Variance')
for ele in dataFileNames:
    acceleration = np.array(extractor.getAccelerationFromFile(path + ele))
    x = acceleration[:,0]
    y = acceleration[:,1]
    z = acceleration[:,2]
    # print('File: ',path+ele)
    print(np.var(x), np.var(y), np.var(z))#'Variance(x,y,z):',
    # print('Max(x,y,z):', np.max(x), np.max(y), np.max(z))
    # print('Min(x,y,z):', np.min(x), np.min(y), np.min(z))

print('Max')
for ele in dataFileNames:
    acceleration = np.array(extractor.getAccelerationFromFile(path + ele))
    x = acceleration[:,0]
    y = acceleration[:,1]
    z = acceleration[:,2]
    # print('File: ',path+ele)
    # print(np.var(x), np.var(y), np.var(z))'Variance(x,y,z):',
    print(np.max(x), np.max(y), np.max(z))#'Max(x,y,z):',
    # print('Min(x,y,z):', np.min(x), np.min(y), np.min(z))

print('Min')
for ele in dataFileNames:
    acceleration = np.array(extractor.getAccelerationFromFile(path + ele))
    x = acceleration[:,0]
    y = acceleration[:,1]
    z = acceleration[:,2]
    # print('File: ',path+ele)
    # print(np.var(x), np.var(y), np.var(z))'Variance(x,y,z):',
    # print('Max(x,y,z):', np.max(x), np.max(y), np.max(z))
    print(np.min(x), np.min(y), np.min(z)) #'Min(x,y,z):',
