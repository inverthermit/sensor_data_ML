import copy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os.path
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace
# from SimpleNamespace import SimpleNamespace as Namespace
from util.Util import Util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../')
classificationNum = 3
rootDir = '../../'
path = rootDir + 'real_data/'

def getCoordinate():
	train_data = pd.read_csv(path + 'quaternion.csv')

	train_data = train_data[[
		'W',
		'X',
		'Y',
		'Z'
		]]

	qww = train_data['W'] * train_data['W']
	qwx = train_data['W'] * train_data['X']
	qwy = train_data['W'] * train_data['Y']
	qwz = train_data['W'] * train_data['Z']
	qxy = train_data['X'] * train_data['Y']
	qxz = train_data['X'] * train_data['Z']
	qyz = train_data['Y'] * train_data['Z']

	orientation = list()
	orientation.append(2 * (qww - 0.5 + train_data['X'] * train_data['X']))
	orientation.append(2 * (qxy + qwz))
	orientation.append(2 * (qxz - qwy))
	orientation.append(2 * (qxy - qwz))
	orientation.append(2 * (qww - 0.5 + train_data['Y'] * train_data['Y']))
	orientation.append(2 * (qyz + qwx))
	orientation.append(2 * (qxz + qwy))
	orientation.append(2 * (qyz - qwx))
	orientation.append(2 * (qww - 0.5 + train_data['Z'] * train_data['Z']))

	coordinate = pd.DataFrame({'X':-1 * orientation[0].as_matrix(), 'Y':-1 * orientation[3].as_matrix(), 'Z':-1 * orientation[6].as_matrix()})

	return coordinate

coordinate = getCoordinate()
step = int(len(coordinate)/6)
index = 0

while index + step < len(coordinate):
	sub_coordinate = coordinate[index: index + step]
	print(index, index+step)
	plt.plot(sub_coordinate['Z'], sub_coordinate['Y'] )#,marker = 'o', linestyle = '-'
	# plt.xlim(-1.1,1.1)
	# plt.ylim(-1.1,1.1)
	plt.show()
	index += step

# print(coordinate)
# plt.plot(coordinate['Z'], coordinate['Y'],marker = 'o', linestyle = '-')
# # print(plt.xlim(), plt.ylim())
# plt.show()
