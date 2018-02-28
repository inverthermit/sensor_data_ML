import os.path
import sys
sys.path.append('../')
from types import SimpleNamespace as Namespace

from util.Util import Util
from os.path import isfile, join
from os import listdir
root_path = '../../'
delete_folder_keys = ['path_chunked_json_folder','image_save_folder','pic_to_np_array'
,'clustering_result_csv_folder','labeled_csv']
for folder in delete_folder_keys:
	path = root_path+Util.getConfig(folder)
	files = [f for f in listdir(path) if isfile(join(path, f))]
	for file in files:
		if(file!='example.csv'):
			os.remove(path +file,dir_fd = None)
