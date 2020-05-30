import random
import numpy as np 
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
from chinese import ChineseAnalyzer

###################################################################################
#                                                                                 #
#       DATASET EXTRACTION                                                        #
#                                                                                 #
###################################################################################
# 4664 labeled events 
# For the labels, the value is 1 if the event is a rumor, and is 0 otherwise. 
# The content of all the posts in are in json format (timestamp and text)
# where each file is named event_id.json, corresponding to individual event


def extract_dataset(event_ids_list, labels_list, filename):
    # fill event ids ans labels list.
    dataset = open(filename, "r")
    lines = dataset.readlines()
    i=0
    maxlen=0
    for line in lines:
        elems = line.split() 
        if len(elems)-2 > maxlen:
            maxlen = len(elems)-2          
    for line in lines:
        elems = line.split() 
        event_id = elems[0] # 1st elem is the event id
        label= elems[1]     # 2nd elem is the label
        event_ids_list.append(event_id[4:len(event_id)])
        labels_list.append(label[6])
        i=i+1
    dataset.close()
    return maxlen
    
event_ids=[]
labels=[]
# extract event ids and event labels from Weiboo.txt

maxposts = extract_dataset(event_ids,labels,"Weibo.txt")

###################################################################################
#                                                                                 #
#       DATASET PROCESSING  (splitting & timestamps extraction)                   #
#                                                                                 #
###################################################################################

### Training/ Test/ validation data splitting
n_ev=4664
N_train = int(0.7 * len(event_ids))
N_val = int((n_ev-N_train)*0.5)
N_test = n_ev-N_train-N_val

labels_train= labels[0:N_train]
event_ids_train = event_ids[0:N_train]

labels_test= labels[N_train:N_train+N_test]
event_ids_test = event_ids[N_train:N_train+N_test]

labels_val= labels[N_train+N_test:]
event_ids_val = event_ids[N_train+N_test:]  


RNN_data_test = []
RNN_data_train = []
RNN_data_val = []

  
## Load pickled text files      
with open("RNN_data_test.txt", "rb") as fp:   # Unpickling
          RNN_data_test=(pickle.load(fp))        
with open("RNN_data_val.txt", "rb") as fp:   # Unpickling
          RNN_data_val=(pickle.load(fp))
with open("RNN_data_train.txt", "rb") as fp:   # Unpickling
          RNN_data_train=(pickle.load(fp))        

N = 10

###################################################################################
#                                                                                 #
#       NETWORK DATA PREPARATION                                                  #
#                                                                                 #
################################################################################### 

## Shuffling
labels= np.array(labels, dtype=int)
# Random data shuffling
reshuffled_labels = labels
reshuffled_data = RNN_data_train.copy()
reshuffled_data.extend(RNN_data_test)
reshuffled_data.extend(RNN_data_val)
temp = list(zip(reshuffled_labels, reshuffled_data))

random.shuffle(temp)
reshuffled_labels, reshuffled_data = zip(*temp)
reshuffled_labels = np.array(reshuffled_labels)

labels_train= reshuffled_labels[0:N_train]
labels_test= reshuffled_labels[N_train:N_train+N_test]
labels_val= reshuffled_labels[N_train+N_test:]

RNN_data_train = [sublist for sublist in reshuffled_data[0:N_train]]
RNN_data_test = [sublist for sublist in reshuffled_data[N_train:N_train+N_test]]
RNN_data_val = [sublist for sublist in reshuffled_data[N_train+N_test:]]

## Padding the RNN sequences
k = 2500 #Vocabulary size
maxNrIntervals=N

new_rnn_train = []
new_rnn_test = []
new_rnn_val = []
# Processing Training Data
for event in RNN_data_train:
    new_event = []
    for interval in event: 
        if isinstance(interval, int):
            interval = [interval]
        kInterval = sorted(interval, reverse=True)[:k]
        kInterval.extend([0]*(k-len(kInterval))) #append the interval with zeros until it has a length of k
        new_event.append(kInterval[0:k])
        if len(new_event) == maxNrIntervals: break
    while len(new_event) < maxNrIntervals:
        new_event.append([0]*k) #append the event with intervals of zeros until it has a length of maxNrIntervals
    new_rnn_train.append(new_event)

for event in RNN_data_test:
    new_event = []
    for interval in event: 
        if isinstance(interval, int):
            interval = [interval]
        kInterval = sorted(interval, reverse=True)[:k]
        kInterval.extend([0]*(k-len(kInterval))) #append the interval with zeros until it has a length of k
        new_event.append(kInterval[0:k])
        if len(new_event) == maxNrIntervals: break
    while len(new_event) < maxNrIntervals:
        new_event.append([0]*k) #append the event with intervals of zeros until it has a length of maxNrIntervals
    new_rnn_test.append(new_event)
    
for event in RNN_data_val:
    new_event = []
    for interval in event: 
        if isinstance(interval, int):
            interval = [interval]
        kInterval = sorted(interval, reverse=True)[:k]
        kInterval.extend([0]*(k-len(kInterval))) #append the interval with zeros until it has a length of k
        new_event.append(kInterval[0:k])
        if len(new_event) == maxNrIntervals: break
    while len(new_event) < maxNrIntervals:
        new_event.append([0]*k) #append the event with intervals of zeros until it has a length of maxNrIntervals
    new_rnn_val.append(new_event)
 
## One-hot encoding of the labels
labels_train =np.array(labels_train)
labels_test = np.array(labels_test)
labels_val =np.array(labels_val)
#Convert labels to one-hot vector
labels_train_onehot = np.zeros((labels_train.shape[0],2))
for indx in range(labels_train.shape[0]):
    labels_train_onehot[indx,int(labels_train[indx])] = 1
    
labels_test_onehot = np.zeros((labels_test.shape[0],2))
for indx in range(labels_test.shape[0]):
    labels_test_onehot[indx,int(labels_test[indx])] = 1
    
labels_val_onehot = np.zeros((labels_val.shape[0],2))
for indx in range(labels_val.shape[0]):
    labels_val_onehot[indx,int(labels_val[indx])] = 1
    

RNN_data_train=np.array(new_rnn_train)
RNN_data_test=np.array(new_rnn_test)
RNN_data_val=np.array(new_rnn_val)

with open('RNN_data_train.npy', 'wb') as f:
    np.save(f, RNN_data_train)

with open('RNN_data_test.npy', 'wb') as f:
    np.save(f, RNN_data_test)

with open('RNN_data_val.npy', 'wb') as f:
    np.save(f, RNN_data_val)

with open('labels_train_onehot.npy', 'wb') as f:
    np.save(f, labels_train_onehot)

with open('labels_test_onehot.npy', 'wb') as f:
    np.save(f, labels_test_onehot)

with open('labels_val_onehot.npy', 'wb') as f:
    np.save(f, labels_val_onehot)

