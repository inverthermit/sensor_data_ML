import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import json
import copy
from types import SimpleNamespace as Namespace
from feature.FeatureExtractor import FeatureExtractor
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor
from util.TimeUtil import TimeUtil
from util.Util import Util

class ProcessFeatureExtractor (SimpleFeatureExtractor):
    """
    Input all the parameters and tag the labels to the data.
    e.g. tagProcessData('gas_flow', 50, 20*60000, 30*60000, originData)
    """

    """lance_flow, mount_type, imu_position, bath_start, immersion"""
    """Trial, Tip, Lance Flow, NGIMU Mount Type,
    NGIMU Position, Bath Start Depth, Immersion,
    Data_Points, Tap Time, Tap Depth	Fill Time, Fill Depth,
    Change of Flow T1, Lance Flow After Change,
    Change of Flow T2, Lance Flow After Change,
    Lance Movement Time, Lance Movement**
    Stop Time
    """

    """Use JSON for this"""
    def tagProcessData(self, dataframe, trailName, returnDataFrame = True):
        if '.json' in trailName:
            trailName = trailName.replace('.json','')
        rootDir = '../../'
        with open(rootDir + Util.getConfig('trials_info_path')) as json_data_file:
            data = json.load(json_data_file)
        trialInfoObj = None
        for ele in data:
            if ele['Trial'] == trailName:
                trialInfoObj = ele
                break

        """Check if the trail exists"""
        if trialInfoObj == None:
            raise ValueError('Trail ' , trailName ,'has no info in trail_info_file. Please input another trail.')
            return

        df = dataframe
        length = len(df['timeStamp'])
        fileStartTimeStamp = df.iloc[0]['timeStamp']
        """1. Set initial data(flow/mount type)"""
        #transfer strings to int
        df['ngimu_mount_type'] = [sum(bytearray(trialInfoObj['NGIMU Mount Type'],'ascii'))] * length
        df['ngimu_position'] = [trialInfoObj['NGIMU Position']] * length
        df['bath_start_depth'] = [trialInfoObj['Bath Start Depth']] * length
        df['immersion'] = [trialInfoObj['Immersion']] * length
        initial_lance_flow = [trialInfoObj['Lance Flow']] * length

        """2. Set changed data"""
        if(trialInfoObj['Change of Flow T1']!=''):
            #[minute, second]
            flow_change_time1_arr = (list(map(int, (trialInfoObj['Change of Flow T1'].split(':')))))
            flow_change_time1 = fileStartTimeStamp + TimeUtil.getMillisecondFromMinute(
                second = flow_change_time1_arr[1], minute= flow_change_time1_arr[0] )

            flow_change_time2_arr = (list(map(int, (trialInfoObj['Change of Flow T2'].split(':')))))
            flow_change_time2 = fileStartTimeStamp + TimeUtil.getMillisecondFromMinute(
                second = flow_change_time2_arr[1], minute= flow_change_time2_arr[0] )
            index1 = -1
            index2 = -1
            for index, row in df.iterrows():
                if(index1 == -1 and row['timeStamp']>= flow_change_time1):
                    index1 = index
                if(index2 == -1 and row['timeStamp']>= flow_change_time2):
                    index2 = index
                    break
            initial_lance_flow[index1:index2] = [trialInfoObj['Lance Flow After Change T1']]*(index2-index1)
            initial_lance_flow[index2:] = [trialInfoObj['Lance Flow After Change T2']]* (length - index2)
            df['lance_flow'] = initial_lance_flow
        # df1 = df[['ngimu_mount_type','ngimu_position', 'bath_start_depth',
        #         'immersion', 'lance_flow']]
        if returnDataFrame:
            return df

        return df1.as_matrix().tolist()



"""end of file"""
