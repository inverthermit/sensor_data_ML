import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import json
import copy
from types import SimpleNamespace as Namespace
from feature.FeatureExtractor import FeatureExtractor
from feature.SimpleFeatureExtractor import SimpleFeatureExtractor
from util.TimeUtil import TimeUtil

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
    def tagProcessData(self, trialInfoObj, data):

        trialInfoStr = """{
                            "Trial": "4c",
                            "Tip": "IV-2",
                            "Lance Flow": 50,
                            "NGIMU Mount Type": "A",
                            "NGIMU Position": -0.090,
                            "Bath Start Depth": 0.08,
                            "Immersion": 0.04,
                            "Data_Points": 21587,
                            "Tap Time": "",
                            "Tap Depth": "",
                            "Fill Time": "",
                            "Fill Depth": "",
                            "Change of Flow T1": "5:05",
                            "Lance Flow After Change T1": 100,
                            "Change of Flow T2": "10:00",
                            "Lance Flow After Change T2": 50,
                            "Lance Movement Time": "",
                            "Lance Movement**": "",
                            "Stop Time": "12:00"
                          }""";
        trialInfoObj = json.loads(trialInfoStr)
        # print(trialInfoObj['Tip'])
        df = pd.DataFrame({'timeStamp': data[:,3], 'x': data[:,0], 'y': data[:,1], 'z': data[:,2], 'label': data[:,4]})
        df = df[['timeStamp','x', 'y', 'z', 'label']]



        startTime = data[0][0]
        length = len(df['timeStamp'])
        fileStartTimeStamp = df.iloc[0]['timeStamp']
        # print(df.iloc[0]['timeStamp'])
        """1. Set initial data(flow/mount type)"""
        #transfer strings to int
        df['ngimu_mount_type'] = [hash(trialInfoObj['NGIMU Mount Type'])%100] * length
        df['ngimu_position'] = [trialInfoObj['NGIMU Position']] * length
        df['bath_start_depth'] = [trialInfoObj['Bath Start Depth']] * length
        df['immersion'] = [trialInfoObj['Immersion']] * length
        initial_lance_flow = [trialInfoObj['Lance Flow']] * length

        """2. Set changed data"""

        #[minute, second]
        flow_change_time1_arr = (list(map(int, (trialInfoObj['Change of Flow T1'].split(':')))))
        flow_change_time1 = fileStartTimeStamp + TimeUtil.getMillisecondFromMinute(
            second = flow_change_time1_arr[1], minute= flow_change_time1_arr[0] )

        flow_change_time2_arr = (list(map(int, (trialInfoObj['Change of Flow T2'].split(':')))))
        flow_change_time2 = fileStartTimeStamp + TimeUtil.getMillisecondFromMinute(
            second = flow_change_time2_arr[1], minute= flow_change_time2_arr[0] )
        # print(flow_change_time1_arr,flow_change_time2_arr)
        index1 = -1
        index2 = -1
        for index, row in df.iterrows():
            if(index1 == -1 and row['timeStamp']>= flow_change_time1):
                index1 = index
            if(index2 == -1 and row['timeStamp']>= flow_change_time2):
                index2 = index
                break
        # print(index1,index2)
        # print(len(initial_lance_flow[index1:index2]),len([trialInfoObj['Lance Flow After Change T1']]*(index2-index1)))
        # print(len(initial_lance_flow[index2:]),len([trialInfoObj['Lance Flow After Change T2']]* (length - index2)))
        initial_lance_flow[index1:index2] = [trialInfoObj['Lance Flow After Change T1']]*(index2-index1)
        initial_lance_flow[index2:] = [trialInfoObj['Lance Flow After Change T2']]* (length - index2)
        # print(len(initial_lance_flow),length)
        df['lance_flow'] = initial_lance_flow
        print(df)



"""end of file"""
