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
import pandas as pd

sys.path.append('../')
rootDir = '../../'
csv_data = pd.read_csv(rootDir+Util.getConfig('path_origin_csv_file'))
save_path = rootDir+Util.getConfig('path_chunked_json_folder')
# print(csv_data)
csv_data['timestamp'] = (csv_data['Time (s)']*1000).astype(int)
# print(csv_data)
train_data = csv_data[[
    'W',
    'X',
    'Y',
    'Z',
    'timestamp'
    ]].values.tolist()[(int(len(csv_data)/6)):]
# print(train_data)

arr = np.array_split(train_data, Util.getConfig('number_of_chunked_data'))
# print(arr[0])
for i in range(Util.getConfig('number_of_chunked_data')):
    file_data = {}

    data = []
    for ele in arr[i].tolist():
        row = {}
        row['timestamp'] = int(ele[4])
        row['event'] = {'variable':'quaternion','content':ele}
        data.append(row)
    """Save to file 'kcm'+'i'+'.json'"""
    file_data['data'] = data
    json_data = json.dumps(file_data)
    # print(json_data)
    output_file = open(save_path+'kcm'+ str(i)+'.json', "w")
    output_file.write(json_data)
    output_file.close()



# print(train_data)
