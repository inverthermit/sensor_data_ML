import copy
import os.path
from sklearn.externals import joblib
import sys
import random
from constant import *
import numpy as np
from SimpleNamespace import SimpleNamespace as Namespace
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor
from feature.TimeSeriesFeatureExtractor import TimeSeriesFeatureExtractor
from feature.FeatureTransformation import FeatureTransformation

class RunningExperiment():

	def experiment(self, exp_num, binary_multiclass, feature_used, if_scaling, if_rotation):
		s = 'Running experiment ' + str(exp_num) + ': '
		if binary_multiclass == Binary_Multi_Class.BINARY:
			s += 'binary classification, '
		else:
			s += 'multi-class classification, '
		if feature_used == Feature_Used.FULL:
			s += 'using feature (x,y,z,mean,std), '
		else:
			s += 'using feature (x,y,z), '
		if if_scaling == Scaling_Rotation.YES:
			s += 'with scaling, '
		else:
			s += 'without scaling, '
		if if_rotation == Scaling_Rotation.YES:
			s += 'with rotation.'
		else:
			s += 'without rotation.'
		print(s)


		path = '../../data/'
		sys.path.append('../')
		extractor = TimeSeriesFeatureExtractor()
		featureTransformer = FeatureTransformation()
		trainFileNames = None
		labels = None
		test_labels = None
		result_file_name = 'train_file_' + str(exp_num) + '.npz'
		result_test_file_name = 'test_file_binary.npz'

		if binary_multiclass == Binary_Multi_Class.BINARY:
			trainFileNames = ['0a.json','0b.json']
			testFileNames = ['4a.json','3f.json']
			labels = [0,1]
			test_labels = [0,1]
		else:
			trainFileNames = ['1b.json','4e.json','0b.json','4b.json','4c.json']
			result_test_file_name = 'test_file_multi.npz'
			testFileNames = ['4a.json', '4f.json', '3f.json', '0c.json', '2b.json']
			labels = [0,1,2,4,5]
			test_labels = labels

		self.load_data(trainFileNames, result_file_name, labels, extractor, path)
		self.load_data(testFileNames, result_test_file_name, test_labels, extractor, path)
		fileContent = np.load(path + result_file_name)
		data = fileContent['data']

		fileContent = np.load(path + result_test_file_name)
		test_data = fileContent['data']

		x_train = None
		y_train = None
		x_test = None
		y_test = None

		if if_rotation == Scaling_Rotation.YES:
			data = featureTransformer.rotateYZAxis(data)
		if if_scaling == Scaling_Rotation.YES:
			data = featureTransformer.scaleYZAxis(data)
			test_data = featureTransformer.scaleYZAxis(test_data)

		if feature_used == Feature_Used.FULL:
			train_no_nan = extractor.insertRollingFeatures(data, window = 350)
			rand_data = np.array(copy.deepcopy(train_no_nan))
			random.shuffle(rand_data)
			x_train = rand_data[:,1:10]
			y_train = rand_data[:,10]

			train_no_nan = extractor.insertRollingFeatures(test_data, window = 350)
			rand_data = np.array(copy.deepcopy(train_no_nan))
			random.shuffle(rand_data)
			x_test = rand_data[:,1:10]
			y_test = rand_data[:,10]
		else:
			random.shuffle(data)
			x_train = data[:,0:3]
			y_train = data[:,4]

			random.shuffle(test_data)
			x_test = test_data[:,0:3]
			y_test = test_data[:,4]


		# X_rand = x_train
		# y_rand = y_train

		# heldout_len = int(len(data)*0.8)
		# train_data = data[:heldout_len]
		# test_data = data[heldout_len:]

		# x_train = X_rand[:heldout_len]
		# y_train = y_rand[:heldout_len]
		# x_test = X_rand[heldout_len:]
		# y_test = y_rand[heldout_len:]


		# if if_rotation == Scaling_Rotation.YES:
		# 	train_data = featureTransformer.rotateYZAxis(train_data)

		# if feature_used == Feature_Used.FULL:
		# 	x_train = train_data[:,1:10]
		# 	y_train = train_data[:,10]
		# 	x_test = test_data[:,1:10]
		# 	y_test = test_data[:,10]
		# else:
		# 	x_train = train_data[:,0:3]
		# 	y_train = train_data[:,4]
		# 	x_test = test_data[:,0:3]
		# 	y_test = test_data[:,4]


		numTree = 9
		"""Random Forest"""
		from sklearn.ensemble import RandomForestClassifier
		rf_model = RandomForestClassifier(n_estimators=numTree, n_jobs = 8)
		model = rf_model
		print('Random Forest(',numTree,'):')

		model.fit(x_train,y_train)
		print('Training score: ',model.score(x_train,y_train))
		print('Testing score: ', model.score(x_test,y_test))

		from sklearn.metrics import classification_report
		y_true = y_test
		y_pred = model.predict(x_test)
		print(labels)
		if binary_multiclass == Binary_Multi_Class.BINARY:
			print(classification_report(y_true, y_pred, target_names = ['0','1']))
		else:
			print(classification_report(y_true, y_pred, target_names = ['0','1','2','4','5']))

		modelName = str(exp_num) + '.pkl'

		model = joblib.dump(model, modelName)
		model = joblib.load(modelName)
		print('Model ' + str(exp_num) + ' saved.')



	def load_data(self, file_name, result_file_name, labels, extractor, path):
		resultFileName = result_file_name
		extractor.saveSimpleFeaturedData(path, file_name, labels, resultFileName)
