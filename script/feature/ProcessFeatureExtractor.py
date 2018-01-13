import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import json
import copy
from types import SimpleNamespace as Namespace
from feature.FeatureExtractor import FeatureExtractor
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor

class ProcessFeatureExtractor (SimpleFeatureExtractor):
    """
    Input all the parameters and tag the labels to the data.
    e.g. tagProcessData(['gas_flow', 'mount_type', 'water_amount'], [50, 'C', 0.23], 20*60000, 30*60000, originData)
    """
    def tagProcessData(self, attrName, value, startTime, endTime, originData):
