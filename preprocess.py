"""
File description : Preprocessing of the data


"""


import numpy as np
import json
import pickle
import sys

#Tools import
import processing.extract_dataset as dat
import processing.time_series_const as time
import processing.post_text_preprocess as pro
import processing.tfidf as tfidf

#Visualization imports
import analysis.eventEvolution as ev
import analysis.plotAccuracy as pltA

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
event_related_posts = dat.extract_dataset(event_ids,labels,"Weibo.txt",n_ev)

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


## load event time series txt files (to save time)

with open("Data/N30/training_event_time_series_tfidf.txt", "rb") as fp:   # Unpickling
          rnn_data_train=(pickle.load(fp))
with open("Data/N30/test_event_time_series_tfidf.txt", "rb") as fp:   # Unpickling
          rnn_data_test=(pickle.load(fp))


# Keep only the K-most important score per interval
# pad the number of intervals (each event need same number of intervals)
# put each event in a numpy array

#Reshuffling the data to get a more even distribution of rumor and non-rumor event across training and test data

import random
reshuffled_labels = labels
reshuffled_data = rnn_data_train.copy()
reshuffled_data.extend(rnn_data_test)
temp = list(zip(reshuffled_labels, reshuffled_data))

random.shuffle(temp)
reshuffled_labels, reshuffled_data = zip(*temp)
reshuffled_labels = np.array(reshuffled_labels)

labels_train= reshuffled_labels[N_test:reshuffled_labels.shape[0],:]
labels_test= reshuffled_labels[0:N_test,:]

rnn_data_train = [sublist for sublist in reshuffled_data[N_test:]]
rnn_data_test = [sublist for sublist in reshuffled_data[0:N_test]]

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(labels_train)
labels_train = enc.transform(labels_train).toarray()
enc.fit(labels_test)
labels_test = enc.transform(labels_test).toarray()


print(labels_train.shape)
print(labels_test.shape)


k = 2500 #the number of tf.idf values sorted in descending order we will keep for each interval
maxNrIntervals = 30 #equivalent to N value

print(maxNrIntervals)
new_rnn_train = []
new_rnn_test = []
#Processing Training Data
print("Start for train")
for idx, event in enumerate(rnn_data_train):
    sys.stdout.write("Progress: %d%%   \r" % (idx/len(rnn_data_train)*100) )
    sys.stdout.flush()
    new_event = []
    for interval in event:
        highestVals = (-np.array(interval)).argsort()[:k]
        kInterval = []
        for i, val in enumerate(interval):
            if (i in highestVals):
               kInterval.append(val)
        kInterval.extend([0]*(k-len(kInterval))) #append the interval with zeros until it has a length of k
        new_event.append(kInterval)
        if len(new_event) == maxNrIntervals: break
    while len(new_event) < maxNrIntervals:
        new_event.append([0]*k) #append the event with intervals of zeros until it has a length of maxNrIntervals
    new_rnn_train.append(new_event)

#Processing Test Data
print("Start for test")
for idx, event in enumerate(rnn_data_test):
    sys.stdout.write("Progress: %d%%   \r" % (idx/len(rnn_data_test)*100) )
    sys.stdout.flush()
    new_event = []
    for interval in event:
        highestVals = (-np.array(interval)).argsort()[:k]
        kInterval = []
        for i, val in enumerate(interval):
            if (i in highestVals):
               kInterval.append(val)
        kInterval.extend([0]*(k-len(kInterval))) #append the interval with zeros until it has a length of k
        new_event.append(kInterval)
        if len(new_event) == maxNrIntervals: break
    while len(new_event) < maxNrIntervals:
        new_event.append([0]*k) #append the event with intervals of zeros until it has a length of maxNrIntervals
    new_rnn_test.append(new_event)

new_rnn_train = np.array(new_rnn_train) #convert the standard python lists to numpy arrays
norm = np.linalg.norm(new_rnn_train)
new_rnn_train = new_rnn_train/norm
new_rnn_test = np.array(new_rnn_test)
new_rnn_test = new_rnn_test/norm
print(new_rnn_train.shape)
print(new_rnn_test.shape)


# ------------------------ Serializing --------------------------------------------

with open("Data/Preprocessed/train.txt", "wb") as fp:
      pickle.dump(new_rnn_train, fp)
with open("Data/Preprocessed/test.txt", "wb") as fp:
      pickle.dump(new_rnn_test, fp)

with open("Data/Preprocessed/label_train.txt", "wb") as fp:
    pickle.dump(labels_train, fp)
with open("Data/Preprocessed/label_test.txt", "wb") as fp:
    pickle.dump(labels_test, fp)
