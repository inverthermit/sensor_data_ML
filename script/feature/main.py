import numpy as np
import matplotlib.pyplot as plt
import json
from types import SimpleNamespace as Namespace
import sklearn

from simpleFeatureExtractor import SimpleFeatureExtractor

classificationNum = 3
path = '../../data/'
dataFileNames = ['drain.json','Pin hole tip.json','Scallop tip.json']
labels = ['normal', 'hole', 'scallop']
extractor = SimpleFeatureExtractor()

for i in range(len(dataFileNames)):
    fileArray = extractor.getAccelerationFromFile(path+dataFileNames[i])
    print(fileArray[100])

    labeledData = extractor.getLabeledData(fileArray, labels[i])
    print(labeledData[100])
