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
from util.Util import Util

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


        # path = '../../data/'

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

        if binary_multiclass == Binary_Multi_Class.BINARY:
            train_file_names = ['0a.json','0b.json']
            test_file_names = ['4a.json','3f.json']
            labels = [0,1]
        else:
            train_file_names = ['1b.json','4e.json','0b.json','4b.json','4c.json']
            test_file_names = ['4a.json', '4f.json', '3f.json', '0c.json', '2b.json']
            labels = [0,1,2,4,5]

        train_data = list()
        test_data = list()
        for index, label in enumerate(labels):
            train_df = extractor.getSimpleFeaturedData(path + train_file_names[index], label)
            test_df = extractor.getSimpleFeaturedData(path + test_file_names[index], label)
            train_data.append(train_df)
            test_data.append(test_df)

        if if_scaling == Scaling_Rotation.YES:
            print('Scaling the y&z axises...')
            for index in range(len(train_data)):
                train_data[index] = featureTransformer.scaleYZAxis(train_data[index])
            for index in range(len(test_data)):
                test_data[index] = featureTransformer.scaleYZAxis(test_data[index])

        if if_rotation == Scaling_Rotation.YES:
            print('Rotating the y&z axises...')
            dfAll = list()
            for df in train_data:
                rotated_df = featureTransformer.rotateYZAxis(df)
                dfAll = dfAll + rotated_df
            train_data = dfAll

        x_train = None
        y_train = None
        x_test = None
        y_test = None

        if feature_used == Feature_Used.FULL:
            # Adding time series features
            dfAll = None
            for df in train_data:
                df = extractor.insertRollingFeatures(df, window = 350)
                if dfAll is None:
                    dfAll = df
                    continue
                else:
                    dfAll = dfAll.append(df)
            train_data = dfAll
            dfAll = None
            for df in test_data:
                df = extractor.insertRollingFeatures(df, window = 350)
                if dfAll is None:
                    dfAll = df
                    continue
                else:
                    dfAll = dfAll.append(df)
            test_data = dfAll

            train_data = train_data[['x', 'y', 'z',
                    'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z',
                    'Rolling_Std_x','Rolling_Std_y','Rolling_Std_z',
                    'label']]
            test_data = test_data[['x', 'y', 'z',
                    'Rolling_Mean_x','Rolling_Mean_y','Rolling_Mean_z',
                    'Rolling_Std_x','Rolling_Std_y','Rolling_Std_z',
                    'label']]
        else:
            # Using simple features
            dfAll = None
            for df in train_data:
                if dfAll is None:
                    dfAll = df
                    continue
                else:
                    dfAll = dfAll.append(df)
            train_data = dfAll
            dfAll = None
            for df in test_data:
                if dfAll is None:
                    dfAll = df
                    continue
                else:
                    dfAll = dfAll.append(df)
            test_data = dfAll

            train_data = train_data[['x', 'y', 'z','label']]
            test_data = test_data[['x', 'y', 'z','label']]

        train_data = train_data.as_matrix()
        test_data = test_data.as_matrix()

        random.shuffle(train_data)
        random.shuffle(test_data)

        x_train = train_data[:,:-1]
        y_train = train_data[:,-1]
        x_test = test_data[:,:-1]
        y_test = test_data[:,-1]

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

        modelName = tmpPath + str(exp_num) + '.pkl'

        model = joblib.dump(model, modelName)
        model = joblib.load(modelName)
        print('Model ' + str(exp_num) + ' saved.')
