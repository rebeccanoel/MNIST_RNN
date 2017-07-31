'''
Reformat Exp. 1
Show 1 image, 4 timesteps of blank images, evaluate how well net has remembered initial stimulus
This function builds the input images and label data sets that will be used for training/test/validation
'''

import numpy as np
import Const_HyperPara as con
from helper_functions import weightBuilder, biasesBuilder, conv2d, maxPool_2x2, rnn_cell

num_blank_steps = 4

blank_images = np.zeros(shape = [num_blank_steps, con.image_size, con.image_size, con.num_channels], dtype = np.float32)

'''
none_label = np.array(100, dtype = np.float32)
blank_labels = none_label
for n in range(num_blank_steps-1):
	blank_labels = np.concatenate((blank_labels, none_label), axis = 0)
 '''

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, con.image_size, con.image_size, con.num_channels)).astype(np.float32)
	labels = (np.arange(con.num_labels) == labels[:,None]).astype(np.float32)
	exp_data = np.concatenate((dataset[:1],blank_images), axis = 0)
	#exp_labels = np.concatenate((labels[:1],blank_labels), axis = 0)
	return exp_data, labels