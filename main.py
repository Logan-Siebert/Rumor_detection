"""
File description : Early detection of rumor propagations using RNN models.

    1) Data structure : We define an event as a chain of posts relative to
                        a given theme. Data from Weibo microblogging is treaded
                        as follow :
            Weibo.txt + *event_id*.json
            labelled data : event -> Rumor or not (1, 0)


    2) Principle : Training various RNNs configuration for classification problem
                   Evolution of an event through time, sequencial input of k-most
                   important words per timestamp as TF-IDF quantifier.

    3) Analysis :
"""

#Base imports
import tensorflow as tf
import numpy as np
import json
import pickle

#Tools import
import processing.extract_dataset as dat
import processing.time_series_const as time
import processing.post_text_preprocess as pro
import processing.tfidf as tfidf

#Visualization imports
import analysis.eventEvolution as ev
import analysis.wordsEvolutionE as we


#misc
np.set_printoptions(precision=3)

###################################################################################
#                                                                                 #
#       DATASET EXTRACTION                                                        #
#                                                                                 #
###################################################################################
# 4664 labeled events
# For the labels, the value is 1 if the event is a rumor, and is 0 otherwise.
# The content of all the posts in are in json format (timestamp and text)
# where each file is named event_id.json, corresponding to individual event

n_ev = 4664 # number of events
event_ids = np.zeros((n_ev,1),dtype=int) #events ids
labels = np.zeros((n_ev,1),dtype=int) # labels 0 or 1

# Extract labels and corresponding event ids
event_related_posts = dat.extract_dataset(event_ids,labels,"Data/Weibo.txt",n_ev)
print(event_related_posts)

print("Loading posts")
# i=0
# j=0
# for event_id in event_ids:
#     filename = 'Weibo/%d.json' %event_id #event file with corresponding posts
#     with open(filename, 'r') as myfile:
#         data=myfile.read()
#     myfile.close()
#     posts = json.loads(data) #event related posts
#     for post in posts:
#         event_related_posts[i,j]=post['t'] #timestamp of post
#         j=j+1
#     event_related_posts[i]= np.sort(event_related_posts[i])
#     i=i+1
#     j=0

#Split the data into training and testing sets 80/20
N_test = int(0.2 * event_related_posts.shape[0])

# event_related_posts_train = event_related_posts[N_test:event_related_posts.shape[0],:]
# labels_train= labels[N_test:labels.shape[0],:]
# event_ids_train = event_ids[N_test:event_ids.shape[0],:]
#
# event_related_posts_test = event_related_posts[0:N_test,:]
# labels_test= labels[0:N_test,:]
# event_ids_test = event_ids[0:N_test,:]

### load pre-saved arrays for time saving
# event_related_posts_train =np.load('Data/event_related_posts_train_array.npy')
# event_related_posts_test =np.load('Data/event_related_posts_test_array.npy')

###################################################################################
#                                                                                 #
#       EVENTS TIME SERIES CONSTRUCTION                                           #
#                                                                                 #
###################################################################################
N=20; # RNN reference length N
# time_series_train = time.events_time_series(event_related_posts_train,N)
# time_series_test = time.events_time_series(event_related_posts_test,N)

###################################################################################
#                                                                                 #
#       TEXTUAL FEATURES EXTRACTION FOR THE TIME SERIES EVENTS                    #
#                                                                                 #
###################################################################################
# Each json file (name=event_id (=1st post ID)) contains posts texts (chinese)
# TF.IDF on each intervals => inputs to RNN
# read time series file and .json corresponding post texts

rnn_data_train=[] #time series tfidf RRN input data for each event
rnn_data_test=[]

#tfidf event: List of event intervals. Each element is dict
#word:tfidf value. Intervals comprise several words-tfidf score pairs
#according to the time series construction of the posts
#we only keep tfidf scores

# for j in range(event_related_posts_train.shape[0]):
#     print("training event: %d/3732" %(j+1))
#     filename = 'Weibo/%d.json' %event_ids_train[j] #training event
#     with open(filename, 'r') as myfile:
#             data=myfile.read()
#     myfile.close()
#     posts = json.loads(data) #training event related posts
#
#     tfidf_event=[]
#     event = time_series_train[j]
#     for i in range(event.shape[0]): #TFIDF for each interval
#         temp = event[i]
#         post_text = pro.post_text_preprocess(temp,posts)
#         #TF-IDF
#         TF_IDF = tfidf.tfidf(post_text)
#         if(TF_IDF!=[]):
#             tfidf_event.append(TF_IDF)
#     if(tfidf_event!=[]):
#         rnn_data_train.append(tfidf_event)
#     posts=[]
#
# for j in range(event_related_posts_test.shape[0]):
#     print("test event: %d/932" %(j+1))
#     filename = 'Weibo/%d.json' %event_ids_test[j] #training event
#     with open(filename, 'r') as myfile:
#             data=myfile.read()
#     myfile.close()
#     posts = json.loads(data) #training event related posts
#
#     tfidf_event=[]
#     event = time_series_test[j]
#     for i in range(event.shape[0]): #TFIDF for each interval
#         temp = event[i]
#         post_text = pro.post_text_preprocess(temp,posts)
#         #TF-IDF
#         TF_IDF = tfidf.tfidf(post_text)
#         if(TF_IDF!=[]):
#             tfidf_event.append(TF_IDF)
#     if(tfidf_event!=[]):
#         rnn_data_test.append(tfidf_event)
#     posts=[]
#
# with open("Data/training_event_time_series_tfidf.txt", "wb") as fp:
#       pickle.dump(rnn_data_train, fp)
# with open("Data/test_event_time_series_tfidf.txt", "wb") as fp:
#       pickle.dump(rnn_data_test, fp)


# ## load event time series txt files (to save time)
with open("Data/N30/training_event_time_series_tfidf.txt", "rb") as fp:   # Unpickling
          rnn_data_train=(pickle.load(fp))
with open("Data/N30/test_event_time_series_tfidf.txt", "rb") as fp:   # Unpickling
          rnn_data_test=(pickle.load(fp))

# ev.plotScatterTime(rnn_data_train, 30)

we.plotEventsExpecteValue(rnn_data_train)
# ev.plotScatterTime(time_serie)
# Keep only the K-most important score per interval
# pad the number of intervals (each event need same number of intervals)
# put each event in a numpy array




###################################################################################
#                                                                                 #
#       word2vec deeplearning methods                                             #
#                                                                                 #
###################################################################################


"""
The first approach of this project was to solve the classification problem by
encoding the features of the problem in time series, and then using a recurrent
neural network. Another approach would use the doc2vec method, reduction of the
more local word2vec method, allowing to represent one entire post as a feature
vector. 
"""
