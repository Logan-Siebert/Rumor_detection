import json


# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------

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
