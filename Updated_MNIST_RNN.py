'''
Deep MNIST for Experts, adapted to RNN
Adapted from modified code that ran w/ Xavier
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#state size is not specified in Feed Forward MNIST model, chosen as 4 in the RNN example
input_size = 784
state_size = 64
#num_classes also not specified in FF MNIST model, assumed to be 10 for MNIST (10 diff. digits)
num_classes = 10

num_batches = 100
#previously num_steps = 5 - dimension incompatibility in final print statement, changed to 10000 but this makes it back propegate 10000 steps for every iteration, takes too long
num_steps = 5
learning_rate = 0.5
batch_size = 5

#BUILDING MODEL
#x = tf.placeholder(tf.float32, shape=[num_steps,input_size])
#y_ = tf.placeholder(tf.int32, shape=[ num_steps,10])

#batch size, # steps, image size


x = tf.placeholder(tf.float32, shape=[batch_size, num_steps, input_size])
y_ = tf.placeholder(tf.int32, shape=[batch_size, num_steps, 10])

init_state = tf.placeholder([None,state_size])

#init_state = tf.zeros([batch_size,state_size])

#processing input data
rnn_inputs = tf.unstack(x, axis=1)
rnn_inputs = [tf.squeeze(rnn_input) for rnn_input in rnn_inputs]
#rnn_inputs are of shape (?,10)



with tf.variable_scope('rnn_cell'):
	W = tf.get_variable('W', [input_size + state_size, state_size])
	b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))


def rnn_cell(rnn_input, state):
	with tf.variable_scope('rnn_cell', reuse=True):
		W = tf.get_variable('W', [state_size + input_size, state_size])
		b = tf.get_variable('b', [state_size], initializer = tf.constant_initializer(0.0))
	return tf.tanh(tf.matmul(tf.concat([tf.squeeze(rnn_input, axis = 0), state], 1), W) + b)

state = init_state
rnn_outputs = []

for rnn_input in rnn_inputs:
	state = rnn_cell(tf.expand_dims(rnn_input, 0), state)
	rnn_outputs.append(state)
final_state = rnn_outputs[-1]

with tf.variable_scope('softmax'):
	W = tf.get_variable('W', [state_size, num_classes])
	b = tf.get_variable('b', [num_classes], initializer = tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]
y_as_list = tf.unstack(y_, num=num_steps, axis = 0)


#rnn_inputs = [tf.expand_dims(rnn_input, 0) for rnn_input in rnn_inputs]

'''
cell = tf.contrib.rnn.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

with tf.variable_scope('soft', reuse = False):
	A = tf.Variable(tf.zeros([state_size, num_classes]), name = "weights")
#A = tf.get_variable('A', [state_size, num_classes])
	#b has shape (10,)
	b = tf.get_variable('b', [num_classes], initializer = tf.constant_initializer(0.0))
	logits = [tf.matmul(rnn_output, A)  for rnn_output in rnn_outputs]

#prediction = tf.matmul(tf.squeeze(rnn_outputs, axis = 1), A)
prediction = tf.matmul(rnn_outputs, A)
#predictions = [tf.nn.softmax(logit) for logit in logits]
y_as_list = tf.unstack(y_, num=num_steps, axis = 0)
'''

sess.run(tf.global_variables_initializer())

#cross entropy function between target and softmax activation function applied to model's prediction
#initialized as a condensed function 'Softmax_Cross_Entropy_Calc' to simplify Tensorboard Visualization

with tf.name_scope('Softmax_Cross_Entropy_Calc') as scope:
	softmax = [tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit) for logit, label in zip(logits, y_as_list)]
	cross_entropy = tf.reduce_mean(softmax)

#TRAINING MODEL
step_length = 0.5
#train_step returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable"
train_step = tf.train.GradientDescentOptimizer(step_length).minimize(cross_entropy)

for _ in range(1000):
	batch = mnist.train.next_batch(batch_size)
	current_in = []
	[current_in.append(batch[0]) for n in range(num_steps)]
	current_label = []
	[current_label.append(batch[1]) for n in range(num_steps)]
	train_step.run(feed_dict = {x: current_in, y_:current_label, init_state: #matrix of size for training images})


#EVALUATE MODEL

correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, #matrix of different size, with batch size of testing images}))


'''




for _ in range(10000):
	batch = mnist.test.next_batch(5)
	print(accuracy.eval(feed_dict={x: batch[0], y_:batch[1]}))
'''

#BUILDING TENSORBOARD GRAPH
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)

print("Done")
