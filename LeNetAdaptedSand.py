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
import numpy as np
import Const_HyperPara as con
from datetime import datetime
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from LeNetFunction import leNet5
#from reformat_original import reformat
from reform_exp_one import reformat
from helper_functions import weightBuilder, biasesBuilder, conv2d, maxPool_2x2, rnn_cell


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


train_dataset, train_labels = reformat(X_train, y_train)
valid_dataset, valid_labels = reformat(X_validation, y_validation)
test_dataset , test_labels  = reformat(X_test, y_test)
submission_dataset = submission_dataset.as_matrix().reshape((-1, con.image_size, con.image_size, con.num_channels)).astype(np.float32)


def gen_timestep_matrix(data, num_steps):
    expanded_tensor = np.expand_dims(data, axis = -1)
    repeat = np.repeat(expanded_tensor,con.num_steps, axis = -1)
    return repeat


print ('Validation set :', len(valid_dataset), len(valid_labels))
print ('Test set       :', len(test_dataset), len(test_labels))
print ('Submission data:', len(submission_dataset))

del X_train,X_validation,X_test,y_train,y_validation,y_test


#IMAGE SIZE AFTER SUB-SAMPLING
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


final_image_size = output_size_no_pool(con.image_size, con.patch_size, padding='same', conv_stride=2)
print(final_image_size)



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

final_image_size = output_size_pool(input_size=con.image_size, conv_filter_size=5, pool_filter_size=2, padding='SAME', conv_stride=1, pool_stride=2)
print(final_image_size)

#LE-NET5
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


finalImageSize = output_size_pool(input_size=con.image_size, conv_filter_size=con.kernelSize,
                                  pool_filter_size=con.poolFilterSize, padding=con.padding,
                                  conv_stride=con.convStride, pool_stride=con.poolStride)



graph = tf.Graph()
with graph.as_default():

    C1_w = weightBuilder([con.kernelSize,con.kernelSize,1,con.depth1Size],"C1_w")
    C1_b = biasesBuilder([con.depth1Size],"C1_b")
    C2_w = weightBuilder([con.kernelSize,con.kernelSize,con.depth1Size,con.depth2Size],"C2_w")
    C2_b = biasesBuilder([con.depth2Size],"C2_b")
    FC1_w = weightBuilder([finalImageSize*finalImageSize*con.depth2Size,con.FC1HiddenUnit],"FC1_w")
    FC1_b = biasesBuilder([con.FC1HiddenUnit],"FC1_b")
    keep_prob = tf.placeholder(dtype=tf.float32,name="keepProb")
    FC2_w = weightBuilder([con.FC1HiddenUnit,con.FC2HiddenUnit],"FC2_w")
    FC2_b = biasesBuilder([con.FC2HiddenUnit],"FC2_b")
    FC3_w = weightBuilder([con.state_size,con.num_labels],"FC3_w")
    FC3_b = biasesBuilder([con.num_labels],"FC3_b")
    RNN_w = weightBuilder([con.state_size + con.input_size, con.state_size], name = "RNN_Weights")
    RNN_b = biasesBuilder([con.state_size], name = "RNN_Biases")
    #RNN_b = biasesBuilder([con.num_steps], name = "RNN_Biases")

    '''
    init_state_train = tf.zeros([con.batch_size,con.state_size])
    init_state_valid = tf.zeros([4200,con.state_size])
    init_state_test = tf.zeros([4200,con.state_size])
    tf_submission_data = tf.placeholder(tf.float32,shape=(con.num_steps, 28000,con.image_size,con.image_size,con.num_channels))
    init_state_sub = tf.zeros([28000,con.state_size])
    '''
    init_state_train = tf.zeros([con.batch_size,con.state_size])
    init_state_valid = tf.zeros([5,con.state_size])
    init_state_test = tf.zeros([5,con.state_size])
    tf_submission_data = tf.placeholder(tf.float32,shape=(con.num_steps, 28000,con.image_size,con.image_size,con.num_channels))
    init_state_sub = tf.zeros([28000,con.state_size])

    tf_train_dataset   = tf.placeholder(tf.float32,shape=(con.batch_size,con.image_size,con.image_size,con.num_channels,con.num_steps))
    tf_train_labels    = tf.placeholder(tf.float32,shape=(con.batch_size,con.num_labels,con.num_steps))
    #validation data 
    tf_valid_dataset   = tf.constant(gen_timestep_matrix(valid_dataset,con.num_steps))
    #test data
    tf_test_dataset    = tf.constant(gen_timestep_matrix(test_dataset,con.num_steps))
    #submission data
    tf_submission_data = tf.placeholder(tf.float32,shape=(28000,con.image_size,con.image_size,con.num_channels,con.num_steps))


    #TRAINING, VALIDATION, TESTING AND SUBMISSION LOOPS

    tt=tf.unstack(tf_train_dataset, axis = -1)
    state_out_train = []
    logits_train = []
    state_out_train.append(init_state_train)
    for train in tt:
        state_out_tmp, logits_tr = leNet5(train, state_out_train[-1],C1_w, C1_b,C2_w,C2_b,FC1_w,FC1_b,FC2_w,FC2_b,RNN_w,RNN_b,FC3_w,FC3_b,keep_prob)
        state_out_train.append(state_out_tmp)
        logits_train.append(logits_tr)

    
    tv = tf.unstack(tf_valid_dataset, axis = -1)
    state_out_validation = []
    logits_val = []
    state_out_validation.append(init_state_valid)
    for valid in tv:
        state_out_v_tmp, logits_v = leNet5(valid, state_out_validation[-1], C1_w, C1_b,C2_w,C2_b,FC1_w,FC1_b,FC2_w,FC2_b,RNN_w,RNN_b,FC3_w,FC3_b,keep_prob)
        state_out_validation.append(state_out_v_tmp)
        logits_val.append(logits_v)
    

    tte = tf.unstack(tf_test_dataset,axis = -1)
    state_out_test = []
    logits_test = []
    state_out_test.append(init_state_test)
    for test in tte:
        state_out_te_tmp, logits_tst = leNet5(test, state_out_test[-1], C1_w, C1_b,C2_w,C2_b,FC1_w,FC1_b,FC2_w,FC2_b,RNN_w,RNN_b,FC3_w,FC3_b,keep_prob)
        state_out_test.append(state_out_te_tmp)
        logits_test.append(logits_tst)

    ts = tf.unstack(tf_submission_data, axis = -1)
    state_out_sub = []
    logits_sub = []
    state_out_sub.append(init_state_sub)
    for sub in ts:
        state_out_sub_tmp, logits_sb = leNet5(sub, state_out_sub[-1], C1_w, C1_b,C2_w,C2_b,FC1_w,FC1_b,FC2_w,FC2_b,RNN_w,RNN_b,FC3_w,FC3_b,keep_prob)
        state_out_sub.append(state_out_sub_tmp)
        logits_sub.append(logits_sb)

    with tf.name_scope('convolution1') as scope:
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
    with tf.name_scope('loss') as scope:
        print("!!THIS IS LOGITS: ", logits_train)
        print("THIS IS LABELS: ", tf_train_labels)
        loss = []
        training_labels = tf.unstack(tf_train_labels, axis = -1)
        for n in training_labels:
            loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train[-1],labels=n)))
        final_loss = loss[0]
        for r in range(1, len(loss)):
            final_loss += loss[r]
        tf.summary.scalar("cost_function", final_loss)
        
        
    global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(con.learningRate,global_step,200, 0.00001, staircase=True)
    
    #Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=con.learningRate).minimize(loss[-1])
    
    #Prediction for training ,valid,test set
    train_prediction = tf.nn.softmax(logits_train[-1])
    valid_prediction = tf.nn.softmax(logits_val[-1])
    test_prediction  = tf.nn.softmax(logits_test[-1])
    submission_prediction = tf.nn.softmax(state_out_sub[-1])

    #merge all summary tf_valid_dataset
    merged = tf.summary.merge_all()


checkpoint_file =  "./checkpoints/LeNet5checkpoint.ckh"
if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")
if not os.path.exists("./net_train"):
    os.makedirs("./net_train")

num_epochs = 2
#otal)batch divides the number of images in the dataset bt the batch size to give the number of batches
total_batch = int(train_dataset.shape[0]/con.batch_size)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    
    
    now  = str(datetime.now()).replace(" ","_").replace(".","_").replace(":","_")
    print (now)
    train_writer = tf.summary.FileWriter(os.path.join("./net_train", now),session.graph)
    
    print("Initialized")
    for epoch in range(num_epochs):
        avg_cost = 0.
        print ('{:5}|{:15}|{:15}|{:15}'.format( "Step","loss","MiniBatch acc","valid acc"))
        for step in range(total_batch):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * con.batch_size) % (train_labels.shape[0] - con.batch_size)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + con.batch_size), :]
            batch_labels = train_labels[offset:(offset + con.batch_size), :]

            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : gen_timestep_matrix(batch_data,con.num_steps),
                         tf_train_labels : gen_timestep_matrix(batch_labels,con.num_steps),
                         keep_prob:0.5}

            _, l, predictions,summary = session.run([optimizer, loss, train_prediction,merged], feed_dict=feed_dict)

            #append each summary 
            train_writer.add_summary(summary, epoch * total_batch + step)
            # Compute average loss
            avg_cost += l[-1] / total_batch
            
            
            if (step % 200 == 0):
                valid_pred = session.run(valid_prediction,feed_dict={keep_prob:1})
                print ('{:5}|{:15}|{:15}|{:15}'.format("{:d}".format(step),
                                                      "{:.9f}".format(l[-1]),
                                                      "{:.9f}".format(accuracy(predictions, batch_labels)),
                                                      "{:.9f}".format(accuracy(valid_pred, valid_labels))))
        #end for batch
        
        if (epoch+1) % 1 == 0:
            print ("\nEpoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost),"\n")

    
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
                                feed_dict ={tf_submission_data:gen_timestep_matrix(submission_dataset, con.num_steps),keep_prob:1})
    
    # Write predictions to csv file
    results = pd.DataFrame({'ImageId': pd.Series(range(1, len(sub_precition[0]) + 1)),
                            'Label'  : pd.Series(sub_precition[0])})
    results.to_csv('LeNet5Results.csv', index=False)


