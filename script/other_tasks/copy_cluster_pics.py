import copy
import random
import numpy as np
import math
import json
import os.path
import sys
sys.path.append('../')
from SimpleNamespace import SimpleNamespace as Namespace
from shutil import copyfile
from util.Util import Util

sys.path.append('../')
classificationNum = 3
rootDir = '../../'
src_path = rootDir + 'pic_with_filter/'
dest_path = rootDir + 'cluster/'

test_np_array = np.load(rootDir + 'nparray/kmeans7kAll.npz')['data']

for index in range(test_np_array.shape[0]):
	file_name = test_np_array[index][0]
	cluster = test_np_array[index][1]
	copyfile(src_path + file_name, dest_path + cluster + '/' + file_name)
