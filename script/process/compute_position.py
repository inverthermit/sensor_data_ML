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


img_interval = 20
sys.path.append('../')
classificationNum = 3
rootDir = '../../'
path = rootDir + Util.getConfig('trials_folder_path')
tmpPath = rootDir + Util.getConfig('tmp_path')

extractor = TimeSeriesFeatureExtractor()
featureTransformer = FeatureTransformation()

train_file_names = None
test_file_names = None
labels = None


train_file_names = ['1a.json', '1b.json', '1c.json', '1d.json', '1e.json',
					'2a.json', '2b.json', '3a.json', '3b.json', '3c.json',
					'3d.json', '3e.json', '3f.json', '3g.json', '4a.json',
					'4b.json', '4c.json', '4d.json', '4e.json', '4f.json',
					'4g.json', '4h.json']
labels = {'1a.json':0,
		  '1b.json':0,
		  '1c.json':0,
		  '1d.json':0,
		  '1e.json':0,
		  '2a.json':0,
		  '2b.json':1,
		  '3a.json':0,
		  '3b.json':1,
		  '3c.json':1,
		  '3d.json':1,
		  '3e.json':1,
		  '3f.json':1,
		  '3g.json':1,
		  '4a.json':0,
		  '4b.json':1,
		  '4c.json':1,
		  '4d.json':1,
		  '4e.json':1,
		  '4f.json':1,
		  '4g.json':1,
		  '4h.json':0}


train_data = list()
for train_file_name in train_file_names:
	train_df = extractor.getQuaternionData(path + train_file_name, labels[train_file_name])
	train_data.append(train_df)

# dfAll = None
# for df in train_data:
# 	print('data file lenght:' + str(len(df)))
# 	if dfAll is None:
# 		dfAll = df
# 		continue
# 	else:
# 		dfAll = dfAll.append(df)
# train_data = dfAll


######################################################
#######                  Test                  #######
######################################################

# test_data = dict()
# test_data['x'] = [1,4,2,-2]
# test_data['y'] = [2,-1,3,2]
# test_data['z'] = [1,-1,-1,-1]
# test_data['label'] = [1,1,1,1]

# train_data = pd.DataFrame(data = test_data)

for index, train_file_name in enumerate(train_file_names):
# for index in [5]:
	current_data = train_data[index][[
		'w',
		'x',
		'y',
		'z',
		'label'
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

	start = 15000
	end = 15020
	coordinate = pd.DataFrame({'x':-1 * orientation[0].as_matrix(), 'y':-1 * orientation[1].as_matrix(), 'z':-1 * orientation[2].as_matrix()})#[start:end] #slicing

	# fig, ax = plt.subplots()
	# xdata, ydata = [], []
	# ln, = plt.plot([], [], 'r-', animated=True)

	# def init():
	# 	ax.set_xlim(-0.002, 0.004)
	# 	ax.set_ylim(0.00005, 0.00030)
	# 	return ln,

	# def update(frame):
	# 	xdata.append(coordinate['y'][frame])
	# 	ydata.append(coordinate['z'][frame])
	# 	ln.set_data(xdata, ydata)
	# 	return ln,

	# ani = FuncAnimation(fig, update, frames=np.linspace(0,20,20),
	# 					init_func=init, blit=True)

	# def init():
	# 	ax.set_xlim(0, 2*np.pi)
	# 	ax.set_ylim(-1, 1)
	# 	return ln,

	# def update(frame):
	# 	xdata.append(frame)
	# 	ydata.append(np.sin(frame))
	# 	ln.set_data(xdata, ydata)
	# 	return ln,

	# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
	# 					init_func=init, blit=True)

	# plt.plot(coordinate['y'].as_matrix(), coordinate['z'].as_matrix())
	# plt.show()



	for index in range(int(len(coordinate) / img_interval)):

		y = coordinate['y'][index * img_interval : index * img_interval + img_interval].as_matrix()
		z = coordinate['z'][index * img_interval : index * img_interval + img_interval].as_matrix()

		plt.plot(y, z)
		plt.axis('off')
		plt.savefig(tmpPath + train_file_name + str(index) + '.png', dpi = 10)
		plt.clf()

# for image_path in glob.glob(tmpPath + '*.png'):
# 	image = misc.imread(image_path)
# 	print(image.shape)
# 	print(type(image))
# 	print(image)

# im = Image.open(tmpPath + '1.png')
# im_grey = im.convert('L')
# im_array = np.array(im_grey)

# for index, line in enumerate(im_array):
# 	im_array[index,:] = [0 if x == 255 else 1 for x in line]


# print(im_array.shape)
# print(im_array)
