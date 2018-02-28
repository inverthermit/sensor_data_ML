from PIL import Image
import copy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os.path
from os.path import isfile, join
from os import listdir
# from PIL import Imagea

import sys
sys.path.append('../')
from SimpleNamespace import SimpleNamespace as Namespace
from util.Util import Util

sys.path.append('../')
rootDir = '../../'
tmpPath = rootDir + Util.getConfig('image_save_folder')
savePath = rootDir + Util.getConfig('pic_to_np_array')
print(savePath)
all_np_array = dict()

counter = 0

fileNames = [f for f in listdir(tmpPath) if isfile(join(tmpPath, f))]
for fileName in fileNames:

	im = Image.open(tmpPath + fileName)
	# im.thumbnail((64,64), Image.ANTIALIAS)
	# plt.imshow(im)
	# plt.show()
	im_grey = im.convert('L')
	im_array = np.array(im_grey)

	for index, line in enumerate(im_array):
		im_array[index,:] = [0 if x == 255 else 1 for x in line]

	width, height = im_array.shape
	file_name = fileName.split('.')[0]

	if file_name in all_np_array:
		all_np_array[file_name] = np.vstack((all_np_array[file_name], im_array.reshape(1, width * height)))
	else:
		all_np_array[file_name] = im_array.reshape(1, width * height)

	counter += 1
	if counter % 1000 == 0:
		print(counter)

for key in all_np_array:
	print(all_np_array[key].shape)
	np.savez(savePath + key + '.npz', data = all_np_array[key])
