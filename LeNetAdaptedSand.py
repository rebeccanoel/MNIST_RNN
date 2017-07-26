'''
Le Net Adapted
'''

#IMPORT LIBRARY
import pandas as pd
import tensorflow as tf
import numpy as np
import operator
import matplotlib.pyplot as plt
import os 
from datetime import datetime
from tabulate import tabulate
from sklearn.model_selection import train_test_split

#IMPORT DATA
file_path ="./input/train.csv"
data   = pd.read_csv(file_path)

label_name  = "label"

dataset = data.drop(label_name, 1)

labels =data.ix[:,label_name]
X_train, X_test,y_train, y_test = train_test_split(dataset.as_matrix(),
                                                   labels,
                                                   test_size=0.2,
                                                   random_state=0)

X_test,X_validation,y_test,y_validation = train_test_split(X_test,
                                                           y_test,
                                                           test_size=0.5,
                                                           random_state=0)


del labels,data,dataset

#load submission_data
file_path ="./input/test.csv"
submission_dataset = pd.read_csv(file_path)

#RESHAPE DATA
#format array to image format 
image_size = 28
num_labels = 10
num_channels = 1 # grayscale
batch_size = 16

#the following included to bridge RNN and LeNet Code with diff variable names
input_size = image_size**2
num_classes = num_labels
state_size = 2000
num_batches = 2000
num_steps = 2

import numpy as np

def reformat(dataset, labels):
  #dataset = dataset.reshape((-1, num_steps, image_size, image_size, num_channels)).astype(np.float32)
  dataset = dataset.reshape((-1, num_steps, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


#the following created to unstack matricies in the case where num_steps is included in the tensor shape itsself
tr_data, train_labels = reformat(X_train, y_train)
v_data, valid_labels = reformat(X_validation, y_validation)
te_data , test_labels  = reformat(X_test, y_test)
submission_dataset = submission_dataset.as_matrix().reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
#Unstacking in order to incorporate num_steps parameter in above
train_dataset = tf.unstack(tr_data, num = num_steps, axis = 1)
valid_dataset = tf.unstack(v_data, num = num_steps, axis = 1)
test_dataset = tf.unstack(te_data, num = num_steps, axis = 1)
print ('Training set   :', len(train_dataset), len(train_labels))

def gen_timestep_matrix(data, num_steps)
    expanded_tensor = tf.expand_dims(data)
    expand_2 = expanded_tensor
    for n in range(num_steps-1):
        expand_2 = tf.stack(expanded_tensor, expand_2)
    return expand_2

'''
    #Original Input Data Format from Le Net
    tf_train_dataset   = tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))
    tf_train_labels    = tf.placeholder(tf.float32,shape=(batch_size,num_labels))
    #validation data 
    tf_valid_dataset   = tf.constant(valid_dataset)
    #test data
    tf_test_dataset    = tf.constant(test_dataset)
    #submission data
    tf_submission_data = tf.placeholder(tf.float32,shape=(28000,image_size,image_size,num_channels))

train_dataset, train_labels = reformat(X_train, y_train)
valid_dataset, valid_labels = reformat(X_validation, y_validation)
test_dataset , test_labels  = reformat(X_test, y_test)
submission_dataset = submission_dataset.as_matrix().reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
'''

print ('Validation set :', len(valid_dataset), len(valid_labels))
print ('Test set       :', len(test_dataset), len(test_labels))
print ('Submission data:', len(submission_dataset))

del X_train,X_validation,X_test,y_train,y_validation,y_test


#IMAGE SIZE AFTER SUB-SAMPLING
#get final image size 
# Create image size function based on input, filter size, padding and stride
# 2 convolutions only
image_size = 28
def output_size_no_pool(input_size, filter_size, padding, conv_stride):
    if padding == 'same':
        padding = -1.00
    elif padding == 'valid':
        padding = 0.00
    else:
        return None
    output_1 = float(((input_size - filter_size - 2*padding) / conv_stride) + 1.00)
    output_2 = float(((output_1 - filter_size - 2*padding) / conv_stride) + 1.00)
    return int(np.ceil(output_2))

patch_size = 5
final_image_size = output_size_no_pool(image_size, patch_size, padding='same', conv_stride=2)
print(final_image_size)

image_size = 28
# Create image size function based on input, filter size, padding and stride
# 2 convolutions only with 2 pooling
def output_size_pool(input_size, conv_filter_size, pool_filter_size, padding, conv_stride, pool_stride):
    if padding == 'SAME':
        padding = -1.00
    elif padding == 'VALID':
        padding = 0.00
    else:
        return None
    # After convolution 1
    output_1 = (((input_size - conv_filter_size - 2*padding) / conv_stride) + 1.00)
    # After pool 1
    output_2 = (((output_1 - pool_filter_size - 2*padding) / pool_stride) + 1.00)    
    # After convolution 2
    output_3 = (((output_2 - conv_filter_size - 2*padding) / conv_stride) + 1.00)
    # After pool 2
    output_4 = (((output_3 - pool_filter_size - 2*padding) / pool_stride) + 1.00)  
    return int(output_4)

final_image_size = output_size_pool(input_size=image_size, conv_filter_size=5, pool_filter_size=2, padding='SAME', conv_stride=1, pool_stride=2)
print(final_image_size)

#LE-NET5
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

  #leNet5
image_size = 28


kernelSize = 5
depth1Size = 6
depth2Size = 16
num_channels = 1

padding="SAME"
convStride = 1
poolStride = 2
poolFilterSize = 2

FC1HiddenUnit = 360
FC2HiddenUnit = 784

learningRate=1e-4

finalImageSize = output_size_pool(input_size=image_size, conv_filter_size=kernelSize,
                                  pool_filter_size=poolFilterSize, padding=padding,
                                  conv_stride=convStride, pool_stride=poolStride)

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
    print("Trying to concat rnn_input + state: ", rnn_input, " + ", state)
    print("Trying to multiply concat * W: ", tf.concat([rnn_input, state], 1), " * ", W)
    print("Trying to add: matmul + b", tf.matmul(tf.concat([rnn_input, state], 1), W), " + ", b)
    return tf.nn.relu(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

C1_w = weightBuilder([kernelSize,kernelSize,1,depth1Size],"C1_w")
C1_b = biasesBuilder([depth1Size],"C1_b")
C2_w = weightBuilder([kernelSize,kernelSize,depth1Size,depth2Size],"C2_w")
C2_b = biasesBuilder([depth2Size],"C2_b")
FC1_w = weightBuilder([finalImageSize*finalImageSize*depth2Size,FC1HiddenUnit],"FC1_w")
FC1_b = biasesBuilder([FC1HiddenUnit],"FC1_b")
keep_prob = tf.placeholder(dtype=tf.float32,name="keepProb")
FC2_w = weightBuilder([FC1HiddenUnit,FC2HiddenUnit],"FC2_w")
FC2_b = biasesBuilder([FC2HiddenUnit],"FC2_b")
FC3_w = weightBuilder([state_size,num_labels],"FC3_w")
FC3_b = biasesBuilder([num_labels],"FC3_b")
RNN_w = weightBuilder([state_size + input_size, state_size], name = "RNN_Weights")
RNN_b = biasesBuilder([state_size], name = "RNN_Biases")
tf_train_dataset   = tf.placeholder(tf.float32,shape=(num_steps, batch_size,image_size,image_size,num_channels))
tf_train_labels    = tf.placeholder(tf.float32,shape=(num_steps, batch_size,num_labels))
init_state_train = tf.zeros([batch_size,state_size])
tf_submission_data = tf.placeholder(tf.float32,shape=(num_steps, 28000,image_size,image_size,num_channels))
init_state_sub = tf.zeros([28000,state_size])

graph = tf.Graph()
with graph.as_default():
    def leNet5(data,state_in):
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
        print(h_FC2)
        '''
        print("state_in is: ", state_in)

        state_in_shape = state_in.get_shape().as_list()
        number_batches = state_in_shape[0]/batch_size
        print("num_batches is: ", number_batches)

        state_in_split = tf.split(state_in, int(number_batches))
        
        print("this is split state_in:", state_in_split)
    
        '''
        #h_FC2 --> (16,784)
        #state --> (784,2000) (can't concat)
        #for batch in state_in_split:
            #print("batch: ", batch, "state_in_spl")
            #state_out = rnn_cell(h_FC2, RNN_w, RNN_b, state_in)
        state_out = rnn_cell(h_FC2, RNN_w, RNN_b, state_in)

        #FC3
        FC_output = tf.matmul(state_out,FC3_w)+FC3_b
        print("This is FC_Output: ", FC_output)
        #shape is (16, 10)

        #RNN
        print("THIS IS RNN_INPUT", FC_output)
        
        return state_out, FC_output #rnn_cell(rnn_in, RNN_w, RNN_b, state)
    

    tt=tf.unstack(tf_train_dataset)
    state_out = init_state_train
    print("this is train_dataset: ", train_dataset)
    for train in tt:
        print("!!!!! This is train: ", train)
        state_out, logits = leNet5(train, state_out)
    state_out = init_state_valid
    for valid in valid_dataset:
        state_out, logits = leNet5(valid, state_out)
    state_out = init_state_test
    for test in test_dataset:
        state_out, logits = leNet5(test, state_out)

    state_out = init_state_sub
    for sub in submission_data:
        state_out, logits = leNet(sub, state_out)

    with tf.name_scope('convolution1') as scope:
        #weight & biases
        tf.summary.histogram("C1_w",C1_w)
        tf.summary.histogram("C1_b",C1_b)
    
    with tf.name_scope('convolution2') as scope:
        tf.summary.histogram("C2_w",C2_w)
        tf.summary.histogram("C2_w",C2_w)
        
    with tf.name_scope('fullyConct1') as scope:
        tf.summary.histogram("FC1_w",FC1_w)
        tf.summary.histogram("FC1_b",FC1_b)
        
    with tf.name_scope('fullyConct2') as scope:
        tf.summary.histogram("FC2_w",FC2_w)
        tf.summary.histogram("FC2_b",FC2_b)
        
    with tf.name_scope('fullyConct3') as scope:
        tf.summary.histogram("FC3_w",FC3_w)
        tf.summary.histogram("FC3_b",FC3_b)

    #with tf.name_scope('RNN_Layer') as scope:


    


    #train computation
    #state_out, logits = leNet5(tf_train_dataset, init_state_train)

    #
    
    with tf.name_scope('loss') as scope:
        print("THIS IS LOGITS: ", logits)
        print("THIS IS LABELS: ", tf_train_labels)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_train_labels))
        #logits has dim (16, 2000), should be (16, 10)
        tf.summary.scalar("cost_function", loss)
        
        
    global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(learningRate,global_step,200, 0.00001, staircase=True)
    #Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)
    
    #Prediction for training ,valid,test set
    train_prediction = tf.nn.softmax(logits)
    x, valid_output = leNet5(tf_valid_dataset, init_state_valid)
    valid_prediction = tf.nn.softmax(valid_output)
    z, test_out = leNet5(tf_test_dataset, init_state_test)
    test_prediction  = tf.nn.softmax(test_out)
    t, sub_out = leNet5(tf_submission_data, init_state_sub)
    submission_prediction = tf.nn.softmax(sub_out)

    #merge all summary tf_valid_dataset
    merged = tf.summary.merge_all()

#%%time
checkpoint_file =  "./checkpoints/LeNet5checkpoint.ckh"
if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")
if not os.path.exists("./net_train"):
    os.makedirs("./net_train")

num_epochs = 2
total_batch = int(train_dataset.shape[0]/batch_size)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    
    #declare log dir
    now  = str(datetime.now()).replace(" ","_").replace(".","_").replace(":","_")
    print (now)
    train_writer = tf.summary.FileWriter(os.path.join("./net_train", now),session.graph)
    
    print("Initialized")
    for epoch in range(num_epochs):
        avg_cost = 0.
        #print tabulate(tabular_data=[],headers=["Step","loss","MiniBatch acc","valid acc"],tablefmt='orgtbl')
        print ('{:5}|{:15}|{:15}|{:15}'.format( "Step","loss","MiniBatch acc","valid acc"))
        for step in range(total_batch):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data,
                         tf_train_labels : batch_labels,
                         keep_prob:0.5}

            _, l, predictions,summary = session.run([optimizer, loss, train_prediction,merged], feed_dict=feed_dict)

            #append each summary 
            train_writer.add_summary(summary, epoch * total_batch + step)
            # Compute average loss
            avg_cost += l / total_batch
            
            
            if (step % 200 == 0):
                valid_pred = session.run(valid_prediction,feed_dict={keep_prob:1})
                print ('{:5}|{:15}|{:15}|{:15}'.format("{:d}".format(step),
                                                      "{:.9f}".format(l),
                                                      "{:.9f}".format(accuracy(predictions, batch_labels)),
                                                      "{:.9f}".format(accuracy(valid_pred, valid_labels))))
        #end for batch
        
        if (epoch+1) % 1 == 0:
            print ("\nEpoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost),"\n")

    #endFOR epoches
    
    test_pred = session.run(test_prediction,feed_dict={keep_prob:1})
    print("Test accuracy: %.6f%%" % accuracy(test_pred, test_labels))
    
    #save the model
    saver = tf.train.Saver()
    saver.save(session, checkpoint_file)

#PREDICTION ON SUBMISSION DATASET
#make prediction
#restore session
with tf.Session(graph=graph) as session:
    
    saver = tf.train.import_meta_graph(checkpoint_file +'.meta')
    saver.restore(session, checkpoint_file)
    sub_precition = session.run([tf.argmax(submission_prediction, 1)],
                                feed_dict ={tf_submission_data:submission_dataset,keep_prob:1})
    
    # Write predictions to csv file
    results = pd.DataFrame({'ImageId': pd.Series(range(1, len(sub_precition[0]) + 1)),
                            'Label'  : pd.Series(sub_precition[0])})
    results.to_csv('LeNet5Results.csv', index=False)


