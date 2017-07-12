'''
Adding a 3rd layer
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
#sess = tf.InteractiveSession()

import numpy as np

import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from datetime import datetime

#os.system("rm -fr /Users/rebeccanoel/Desktop/Building_RNN/New_Graphs_Here")

input_size = 784
state_size = 200
num_classes = 10
num_batches = 2000
num_steps = 2
batch_size = 100

#BUILDING MODEL
x = tf.placeholder(tf.float32, shape=[num_steps, batch_size, input_size], name = "Input_Images")
y_ = tf.placeholder(tf.int32, shape=[num_steps, batch_size, num_classes], name = "Image_Labels")
init_state = tf.placeholder(tf.float32, shape = [batch_size,state_size], name = "Initial_State")
#init_state = tf.placeholder(tf.float32, shape = [None,state_size], name = "Initial_State")

#processing input data
rnn_inputs = tf.unstack(x, axis=0)

with tf.variable_scope('rnn_cell'):
	#state size = # nodes are included
	W = tf.get_variable('W', [input_size + state_size, state_size])
	b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
	#b = tf.get_variable('b', [state_size], initializer=tf.truncated_normal([state_size], mean = 0.5, stddev = 0.25))

state = init_state
rnn_outputs = []

def rnn_cell(rnn_input, state):
	with tf.variable_scope('rnn_cell', reuse=True):
		W = tf.get_variable('W')#, [state_size + input_size, state_size])
		b = tf.get_variable('b')#, [state_size], initializer = tf.constant_initializer(0.0))
	return tf.nn.relu(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

for rnn_input in rnn_inputs:
	state = rnn_cell(rnn_input, state)
	rnn_outputs.append(state)

with tf.variable_scope('softmax'):
	W2 = tf.get_variable('W', [state_size, num_classes])
	b2 = tf.get_variable('b', [num_classes], initializer = tf.constant_initializer(0.0))
	
logits = [tf.matmul(rnn_output, W2) + b2 for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]
y_as_list = tf.unstack(y_, num=num_steps, axis = 0)



#cross entropy function between target and softmax activation function applied to model's prediction
#initialized as a condensed function 'Softmax_Cross_Entropy_Calc' to simplify Tensorboard Visualization

with tf.name_scope('Softmax_Cross_Entropy_Calc') as scope:
	softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_as_list[-1], logits=logits[-1]) #for label, logit in zip(y_as_list, logits)]
	cross_entropy = tf.reduce_mean(softmax)
	tf.summary.scalar("cross_entropy", cross_entropy)

#TRAINING MODEL
step_length = 0.5
#train_step returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable"
train_step = tf.train.GradientDescentOptimizer(step_length).minimize(cross_entropy,global_step=None,
    var_list=(W, b, W2, b2))
writer = tf.summary.FileWriter('./New_Graphs_Here/run' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')) #sess.graph)

#EVALUATE MODEL
correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  



merged_summaries = tf.summary.merge_all()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(50000):
		batch = mnist.train.next_batch(batch_size)
		current_in = []
		[current_in.append(batch[0]) for n in range(num_steps)]
		current_label = []

		[current_label.append(batch[1]) for n in range(num_steps)]
		ss, _ = sess.run([merged_summaries, train_step],feed_dict = {x:current_in, y_:current_label, init_state:np.zeros([batch_size, state_size])})
		writer.add_summary(ss,i)
		

		#writer.add_summary(sess.run(summary_op))

	    
	summ_accuracy = tf.summary.scalar("accuracy", accuracy)

	accuracy_list = []


	for i in range(1000):
		batch_test = mnist.train.next_batch(batch_size)
		current_in_test = []
		[current_in_test.append(batch_test[0]) for n in range(num_steps)]
		current_label_test = []
		[current_label_test.append(batch_test[1]) for n in range(num_steps)]
		with tf.name_scope("Accuracy_Calc"):
			ss, accur = sess.run( [summ_accuracy, accuracy] , feed_dict={x: current_in_test, y_: current_label_test, init_state: np.zeros([batch_size, state_size])})
			accuracy_list.append(accur)
			writer.add_summary(ss,i)
		if i % 100 ==0:
			print(accur)
	#with tf.name_scope('Scalar_Summaries'):
		

	#print(accur)
	#tf.summary.FileWriter.add_summary(sess.run(summary_op))

'''

with tf.name_scope('summaries'):
 	tf.summary.scalar("accuracy", accur)
 	#tf.summary.image("input_image",current_in, max_outputs = 1)
'''

#CALCULATE AVG ACCURACY
calc_1 = sum(accuracy_list)
final_accur = calc_1/1000
print("Avg accuracy is: ",final_accur)

#BUILDING TENSORBOARD GRAPH
#with tf.Session() as sess:
#writer = tf.summary.FileWriter('./New_Graphs_Yay', sess.graph)
	#writer.add_summary(sess.run(summary_op))




