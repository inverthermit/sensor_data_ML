import copy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import os.path
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from scipy import misc
import glob
from PIL import Image
import sys
sys.path.append('../')
from SimpleNamespace import SimpleNamespace as Namespace
from feature.FeatureTransformation import FeatureTransformation
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor
from feature.TimeSeriesFeatureExtractor import TimeSeriesFeatureExtractor
from util.Util import Util
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from os.path import isfile, join
from os import listdir


img_interval = 500
sys.path.append('../')
classificationNum = 3
rootDir = '../../'
path = rootDir + Util.getConfig('trials_folder_path')
tmpPath = rootDir + Util.getConfig('tmp_path')

extractor = TimeSeriesFeatureExtractor()
featureTransformer = FeatureTransformation()


savePath = rootDir+'real_data/chunked/'

train_file_names = [f for f in listdir(savePath) if isfile(join(savePath, f))]


train_data = list()
for train_file_name in train_file_names:
	train_df = extractor.getQuaternionData(savePath + train_file_name, 0)
	train_data.append(train_df)

y_range = 0
z_range = 0
for index, train_file_name in enumerate(train_file_names):
# for index in [5]:
	current_data = train_data[index][[
		'w',
		'x',
		'y',
		'z',
		]]

	qww = current_data['w'] * current_data['w']
	qwx = current_data['w'] * current_data['x']
	qwy = current_data['w'] * current_data['y']
	qwz = current_data['w'] * current_data['z']
	qxy = current_data['x'] * current_data['y']
	qxz = current_data['x'] * current_data['z']
	qyz = current_data['y'] * current_data['z']

	orientation = list()
	orientation.append(2 * (qww - 0.5 + current_data['x'] * current_data['x']))
	# orientation.append(2 * (qxy + qwz))
	# orientation.append(2 * (qxz - qwy))
	orientation.append(2 * (qxy - qwz))
	# orientation.append(2 * (qww - 0.5 + current_data['y'] * current_data['y']))
	# orientation.append(2 * (qyz + qwx))
	orientation.append(2 * (qxz + qwy))
	# orientation.append(2 * (qyz - qwx))
	# orientation.append(2 * (qww - 0.5 + current_data['z'] * current_data['z']))

	coordinate = pd.DataFrame({'x':-1 * orientation[0].as_matrix(), 'y':-1 * orientation[1].as_matrix(), 'z':-1 * orientation[2].as_matrix()})#[start:end] #slicing


	for index in range(int(len(coordinate) / img_interval)):

		y = coordinate['y'][index * img_interval : index * img_interval + img_interval].as_matrix()
		z = coordinate['z'][index * img_interval : index * img_interval + img_interval].as_matrix()

		y_r = y.max().max() - y.min().min()
		z_r = z.max().max() - z.min().min()
		if y_range < y_r:
			y_range = y_r
		if z_range < z_r:
			z_range = z_r

y_range /=100
z_range /=100

print('y range is ' + str(y_range) + ', z range is ' + str(z_range))

for index, train_file_name in enumerate(train_file_names):
# for index in [5]:
	current_data = train_data[index][[
		'w',
		'x',
		'y',
		'z',
		]]

	qww = current_data['w'] * current_data['w']
	qwx = current_data['w'] * current_data['x']
	qwy = current_data['w'] * current_data['y']
	qwz = current_data['w'] * current_data['z']
	qxy = current_data['x'] * current_data['y']
	qxz = current_data['x'] * current_data['z']
	qyz = current_data['y'] * current_data['z']

	orientation = list()
	orientation.append(2 * (qww - 0.5 + current_data['x'] * current_data['x']))
	# orientation.append(2 * (qxy + qwz))
	# orientation.append(2 * (qxz - qwy))
	orientation.append(2 * (qxy - qwz))
	# orientation.append(2 * (qww - 0.5 + current_data['y'] * current_data['y']))
	# orientation.append(2 * (qyz + qwx))
	orientation.append(2 * (qxz + qwy))
	# orientation.append(2 * (qyz - qwx))
	# orientation.append(2 * (qww - 0.5 + current_data['z'] * current_data['z']))

	coordinate = pd.DataFrame({'x':-1 * orientation[0].as_matrix(), 'y':-1 * orientation[1].as_matrix(), 'z':-1 * orientation[2].as_matrix()})#[start:end] #slicing


	for index in range(int(len(coordinate) / img_interval)):

		y = coordinate['y'][index * img_interval : index * img_interval + img_interval].as_matrix() - coordinate['y'][index * img_interval : index * img_interval + img_interval].mean()
		z = coordinate['z'][index * img_interval : index * img_interval + img_interval].as_matrix() - coordinate['z'][index * img_interval : index * img_interval + img_interval].mean()

		y_r = y.max().max() - y.min().min()
		z_r = z.max().max() - z.min().min()
		print('y range is ' + str(y_r) + ', z range is ' + str(z_r))


		plt.xlim(- y_range / 2, y_range / 2)
		plt.ylim(- z_range / 2, z_range / 2)
		plt.plot(y, z)
		plt.axis('off')
		plt.savefig(tmpPath+'real_data_pic/' + train_file_name + str(index) + '.png', dpi = 10)
		# plt.show()
		plt.clf()
