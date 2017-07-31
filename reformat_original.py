'''
Reformat function
'''
import numpy as np
import Const_HyperPara as con
from helper_functions import weightBuilder, biasesBuilder, conv2d, maxPool_2x2, rnn_cell


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, con.image_size, con.image_size, con.num_channels)).astype(np.float32)
  labels = (np.arange(con.num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels