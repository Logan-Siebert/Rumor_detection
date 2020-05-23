"""
File description : RNN implementation using Tesorflow2 for classification problem
                   Architecture : Sequencial, one LSTM per time interval.
                                  Two output dense unit for cathegorization.

                   Loss function : sparse_categorical_crossentropy
                   Optimizer : Adagrad
"""
#Base imports
import tensorflow as tf
import numpy as np
import json
import pickle
import sys

#Tools import
import processing.extract_dataset as dat
import processing.time_series_const as time
import processing.post_text_preprocess as pro
import processing.tfidf as tfidf

#np.set_printoptions(threshold=sys.maxsize)
###################################################################################
#                                                                                 #
#       Preprocessing                                                             #
#                                                                                 #
###################################################################################

# 4664 labeled events
# For the labels, the value is 1 if the event is a rumor, and is 0 otherwise.
# The content of all the posts in are in json format (timestamp and text)
# where each file is named event_id.json, corresponding to individual event

n_ev = 4664 # number of evenements
#event_ids = np.zeros((n_ev,1),dtype=int) #events ids
#labels = np.zeros((n_ev,1),dtype=int) # labels 0 or 1
event_ids = np.zeros((n_ev,1)) #events ids
labels = np.zeros((n_ev,1)) # labels 0 or 1

# Extract labels and corresponding event ids
event_related_posts = dat.extract_dataset(event_ids,labels,"Data/Weibo.txt",n_ev)
#Split the data into training and testing sets 80/20
N_test = int(0.2 * event_related_posts.shape[0])

#event_related_posts_train = event_related_posts[N_test:event_related_posts.shape[0],:]
labels_train= labels[N_test:labels.shape[0],:]
event_ids_train = event_ids[N_test:event_ids.shape[0],:]

#event_related_posts_test = event_related_posts[0:N_test,:]
labels_test= labels[0:N_test,:]
event_ids_test = event_ids[0:N_test,:]

N=20; # RNN reference length N

rnn_data_train=[] #time series tfidf RRN input data for each event
rnn_data_test=[]

###################################################################################
#                                                                                 #
#       Unpacking data                                                            #
#                                                                                 #
###################################################################################

## load event time series txt files (to save time)
with open("Data/N30/training_event_time_series_tfidf.txt", "rb") as fp:   # Unpickling
          rnn_data_train=(pickle.load(fp))
with open("Data/N30/test_event_time_series_tfidf.txt", "rb") as fp:   # Unpickling
          rnn_data_test=(pickle.load(fp))

# Keep only the K-most important score per interval
# pad the number of intervals (each event need same number of intervals)
# put each event in a numpy array


#Just checking what the max number of tf.idf values (maxK) inside any interval in the data is
maxK = 0
for event in rnn_data_train:
    maxK = max(len(max(event,key=len)),maxK)
print("Largest K value in all intervals = " + str(maxK))
maxNrIntervals = max( len(max(rnn_data_train,key=len)), len(max(rnn_data_test,key=len)))
print("Largest # intervals in a single event = " + str(maxNrIntervals))


k = 5000 #the number of tf.idf values sorted in descending order we will keep for each interval
maxNrIntervals = 30 #equivalent to N value

print(maxNrIntervals)
new_rnn_train = []
new_rnn_test = []

'''
    As the same data will be used either for training and for testing, let us
    build the two structures per event.

    rnn_data_train
    rnn_data_test
'''
#Processing Training Data
for event in rnn_data_train:
    new_event = []
    for interval in event:
        kInterval = sorted(interval, reverse=True)[:k]
        kInterval.extend([0]*(k-len(kInterval))) #append the interval with zeros until it has a length of k
        new_event.append(kInterval)
        if len(new_event) == maxNrIntervals: break
    while len(new_event) < maxNrIntervals:
        new_event.append([0]*k) #append the event with intervals of zeros until it has a length of maxNrIntervals
    new_rnn_train.append(new_event)


#Processing Test Data
for event in rnn_data_test:
    new_event = []
    for interval in event:
        kInterval = sorted(interval, reverse=True)[:k]
        kInterval.extend([0]*(k-len(kInterval))) #append the interval with zeros until it has a length of k
        new_event.append(kInterval)
        if len(new_event) == maxNrIntervals: break
    while len(new_event) < maxNrIntervals:
        new_event.append([0]*k) #append the event with intervals of zeros until it has a length of maxNrIntervals
    new_rnn_test.append(new_event)


#convert the standard python lists to numpy arrays
new_rnn_train = np.array(new_rnn_train)
new_rnn_test = np.array(new_rnn_test)
print("Training set shape : " + str(new_rnn_train.shape))
print("Testing set shape : " + str(new_rnn_test.shape))

###################################################################################
#                                                                                 #
#       Tensorflow RNN                                                            #
#                                                                                 #
###################################################################################

# Selecting computing unit
# gpu:0 -> CPU
# gpu:1 -> Default GPU

tf.device('/gpu:1')
#CUDA_VISIBLE_DEVICES=1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding

model = Sequential()

#model.add(Embedding(input_dim=k, output_dim=100, input_length=maxNrIntervals)) #no idea yet how to get this working for our input
#model.add(LSTM(maxNrIntervals, input_shape=(new_rnn_train.shape[1:]), activation='relu', return_sequences=True))
model.add(LSTM(maxNrIntervals, input_shape=(new_rnn_train.shape[1:])))
#model.add(Dropout(0.2)) #is this really necessary?

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adagrad(lr=0.5, decay=1e-6) #paper uses Adagrad instead of Adam with LR of 0.5 (no mention of decay rate)

model.compile(
    loss='sparse_categorical_crossentropy',
    #this should be mse between the probability distributions of the prediction and ground truth + L2-regularization penalty
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(new_rnn_train,
          labels_train,
          epochs=10,
          validation_data=(new_rnn_test, labels_test))


###################################################################################
#                                                                                 #
#       Analysis                                                                  #
#                                                                                 #
###################################################################################
