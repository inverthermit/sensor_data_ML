import numpy as np
import matplotlib.pyplot as plt
import json
import copy
import os.path
import pandas as pd
from types import SimpleNamespace as Namespace
class QuanternionFeatureExtractor ():

    @staticmethod
    def squeezeWXYZ(df, window = 20):
        wxyz = df[['w','x','y','z']].as_matrix()
        labels = df[['label']].as_matrix().ravel().tolist()
        result = list()
        print(window, len(wxyz)-1)
        for i in range(window,len(wxyz)-1):
            features = wxyz[i-window:i].ravel().tolist();
            features.append(labels[i])

            result.append(features)
        return np.array(result)

# df = pd.DataFrame({'w': [0, 1, 2, 4, 1, 2],'x': [0, 1, 2, 4, 1, 2],
# 'y': [0, 1, 2, 4, 1, 2],'z': [0, 1, 2, 4, 1, 2],'label': [0, 1, 2, 4, 1, 2]})
# print(QuanternionFeatureExtractor.squeezeWXYZ(df, 2))
