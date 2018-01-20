import copy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os.path
from sklearn.externals import joblib
import sys
sys.path.append('../')
from SimpleNamespace import SimpleNamespace as Namespace
from RunningExperiment import RunningExperiment
from constant import *
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor
from feature.TimeSeriesFeatureExtractor import TimeSeriesFeatureExtractor
from util.Util import Util

import pandas as pd

re = RunningExperiment()

for number in range(16):

	'''
	Parsing...
	Decide the arguments for current group experiment
	'''
	num = number
	if_rotation = num % 2
	num = num / 2
	if_scaling = num % 2
	num = num / 2
	feature_used = num % 2
	num = num / 2
	binary_multi_class = num

	re.experiment(number + 1, binary_multi_class, feature_used, if_scaling, if_rotation)