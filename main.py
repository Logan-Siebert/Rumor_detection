import tensorflow as tf
import numpy as np 
import extract_dataset as dat
import time_series_const as time
import post_text_preprocess as pro
import tfidf as tfidf
import json
import pickle

###################################################################################
#                                                                                 #
#       DATASET EXTRACTION                                                        #
#                                                                                 #
###################################################################################
# 4664 labeled events 
# For the labels, the value is 1 if the event is a rumor, and is 0 otherwise. 
# The content of all the posts in are in json format (timestamp and text)
# where each file is named event_id.json, corresponding to individual event

n_ev = 4664 # number of evenements
event_ids = np.zeros((n_ev,1),dtype=int) #events ids
labels = np.zeros((n_ev,1),dtype=int) # labels 0 or 1

# Extract labels and corresponding event ids
event_related_posts = dat.extract_dataset(event_ids,labels,"Weibo.txt",n_ev)

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

#event_related_posts_train = event_related_posts[N_test:event_related_posts.shape[0],:]
labels_train= labels[N_test:labels.shape[0],:]
event_ids_train = event_ids[N_test:event_ids.shape[0],:]

#event_related_posts_test = event_related_posts[0:N_test,:]
labels_test= labels[0:N_test,:]
event_ids_test = event_ids[0:N_test,:]

### load pre-saved arrays for time saving
#event_related_posts_train =np.load('event_related_posts_train_array.npy')
#event_related_posts_test =np.load('event_related_posts_test_array.npy')

###################################################################################
#                                                                                 #
#       EVENTS TIME SERIES CONSTRUCTION                                           #
#                                                                                 #
###################################################################################
N=10; # RNN reference length N
#time_series_train = time.events_time_series(event_related_posts_train,N)
#time_series_test = time.events_time_series(event_related_posts_test,N)
  
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

# for j in range(event_related_posts_test.shape[0]):
#     print("test event: %d/932" %(j+1))
#     filename = 'Weibo/%d.json' %event_ids_test[j] #training event
#     with open(filename, 'r') as myfile:
#             data=myfile.read()
#     myfile.close()
#     posts = json.loads(data) #training event related posts
  
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

# with open("training_event_time_series_tfidf.txt", "wb") as fp:
#       pickle.dump(rnn_data_train, fp)    
# with open("test_event_time_series_tfidf.txt", "wb") as fp:
#       pickle.dump(rnn_data_test, fp)


## load event time series txt files (to save time)
with open("training_event_time_series_tfidf.txt", "rb") as fp:   # Unpickling
          rnn_data_train=(pickle.load(fp))        
with open("test_event_time_series_tfidf.txt", "rb") as fp:   # Unpickling
          rnn_data_test=(pickle.load(fp))
          
          
# Keep only the K-most important score per interval
# pad the number of intervals (each event need same number of intervals)
# put each event in a numpy array

  





    

