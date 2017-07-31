'''
Helper Functions
'''
import tensorflow as tf

padding="SAME"

def weightBuilder(shape,name):
    #shape = [patchSize,patchSize,channel,depth]
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01),name=name)

def biasesBuilder(shape,name):
        #shape = depth  size
    return tf.Variable(tf.constant(1.0, shape=shape),name=name)

def conv2d(x,W,name):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding=padding,name=name)

def maxPool_2x2(x,name):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding=padding,name=name)

def rnn_cell(rnn_input, W, b, state):
    return tf.nn.relu(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)