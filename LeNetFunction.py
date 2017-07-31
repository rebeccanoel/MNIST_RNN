'''
LeNet5 Function
'''
import pandas as pd
import tensorflow as tf
import numpy as np
import operator
import matplotlib.pyplot as plt
import os 
from datetime import datetime
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from helper_functions import weightBuilder, biasesBuilder, conv2d, maxPool_2x2, rnn_cell

def leNet5(data,state_in, C1_w, C1_b,C2_w,C2_b,FC1_w,FC1_b,FC2_w,FC2_b,RNN_w,RNN_b,FC3_w,FC3_b,keep_prob):

        #C1
		h_conv = tf.nn.relu(conv2d(data,C1_w,"conv1")+C1_b)
        
        #S2
		h_pool = maxPool_2x2(h_conv,"pool1")
       
        #C3 
		h_conv = tf.nn.relu(conv2d(h_pool,C2_w,"conv2")+C2_b)

		#S4
		h_pool = maxPool_2x2(h_conv,"pool2")

        #reshape last conv layer 
		shape = h_pool.get_shape().as_list()
		h_pool_reshaped = tf.reshape(h_pool,[shape[0],shape[1]*shape[2]*shape[3]])
        #FULLY CONNECTED NET
        
        #F5
		h_FC1 = tf.nn.relu(tf.matmul(h_pool_reshaped,FC1_w)+FC1_b)
		h_FC1 = tf.nn.dropout(h_FC1, keep_prob=keep_prob)
        

        #F6
		h_FC2 = tf.nn.relu(tf.matmul(h_FC1,FC2_w)+FC2_b)
		h_FC2 = tf.nn.dropout(h_FC2,keep_prob=keep_prob)
        

        #RNN
		state_out = rnn_cell(h_FC2, RNN_w, RNN_b, state_in)

        #FC3
		FC_output = tf.matmul(state_out,FC3_w)+FC3_b
        #shape is (16, 10)

		return state_out, FC_output #rnn_cell(rnn_in, RNN_w, RNN_b, state)