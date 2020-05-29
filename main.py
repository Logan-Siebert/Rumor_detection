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

### Extract timestamps   /!\ Reads lots of json files => save the 3 number arrays files for time saving
#event_related_posts = array(event,posts time stamps)
event_related_posts=np.zeros((n_ev,maxposts))
i=0 
j=0
for event_id in event_ids:
    filename = 'Weibo/%s.json' %event_id #event file with corresponding posts
    with open(filename, 'r') as myfile:
        data=myfile.read()
    myfile.close()
    posts = json.loads(data) #event related posts
    for post in posts:
        event_related_posts[i,j]=post['t'] #timestamp of post
        j=j+1
    event_related_posts[i]= np.sort(event_related_posts[i])
    i=i+1
    j=0

event_related_posts_train=event_related_posts[0:N_train,:]
event_related_posts_test=event_related_posts[N_train:N_train+N_test,:]
event_related_posts_val=event_related_posts[N_train+N_test:,:]

### load pre-saved arrays for time saving
# event_related_posts_train =np.load('event_related_posts_train.npy')
# event_related_posts_test =np.load('event_related_posts_test.npy')
# event_related_posts_val =np.load('event_related_posts_val.npy')

###################################################################################
#                                                                                 #
#       EVENTS TIME SERIES CONSTRUCTION                                           #
#                                                                                 #
###################################################################################

def time_series_const(event_related_posts,RNN_time_series_test,N):
    for i in range(event_related_posts.shape[0]):
        posts=event_related_posts[i]
        mask = posts!=0
        posts = posts[mask]  
        Start = posts[0]
        End = posts[-1]
        Timeline = End - Start
        l = Timeline/ N 
        l_init = l
        count_interv_prev = 0
        while(True):
            intervals=[]
            i=0
            while(i<N+1):
                cond= (posts <l*(i+1)+posts[0])*(posts >= l*i+posts[0])
                u=posts[cond]
                if u.shape[0]!=0:
                    intervals.append(list(u))
                else:
                    intervals.append([])
                i=i+1      
            # continuous interval computation
            continuous_intervals = []
            # each sublist is a continuous super-interval (no empty interval inbetween)
            temp = []
            count = 0
            count_list = [] # list of the continuous interval lengths
            for elem in intervals:
                if elem!=[]:
                    temp.append(elem)
                    count =count+1 
                    if (elem == intervals[-1]):
                        continuous_intervals.append(temp)
                        count_list.append(count)
                else:
                    if (temp != []):
                        continuous_intervals.append(temp)
                        count_list.append(count)
                    temp = []
                    count = 0  
            count_list = np.array(count_list)
            idx_max = np.argmax(count_list)
            max_interval = continuous_intervals[idx_max]  # continuous interval covering the longest time span
            count_interv = count_list[idx_max]  # number of intervals (and so time steps) in max_interval    
            if (count_interv < N and count_interv > count_interv_prev):  # Half l and loop
                l = int(l/2)
                count_interv_prev = count_interv
                max_interval_save = max_interval
            else:
                if l != l_init:
                    max_interval = max_interval_save  # outputs the previous iteration result because no improve
                    break
                else:
                    break      
        RNN_time_series_test.append(max_interval)
       
N=10
time_series_test=[]
time_series_train=[]
time_series_val=[]
time_series_const(event_related_posts_test,time_series_test,N)
time_series_const(event_related_posts_train,time_series_train,N)
time_series_const(event_related_posts_val,time_series_val,N)

###################################################################################
#                                                                                 #
#       TFIDF TEXTUAL FEATURES EXTRACTION                                         #
#                                                                                 #
################################################################################### 
# Each json file (name=event_id (=1st post ID)) contains posts texts (chinese)
# TF.IDF on each intervals => inputs to RNN
# read time series file and .json corresponding post texts  

def features_extract(event_related_posts,event_ids, time_series,RNN_data):
    #TFIDF for each event
    # Each interval seen as a supor-post document
    for j in range(event_related_posts.shape[0]):
        print("event: %d" %(j+1))
        filename = 'Weibo/%d.json' %int(event_ids[j]) #event
        with open(filename, 'r') as myfile:
            data=myfile.read()
        myfile.close()
        posts = json.loads(data) #event related posts
        pre_event=[]
        event = (time_series[j])
        for i in range(len(event)): #TFIDF for each interval
            temp = event[i]
            text = ""
            for k in range(len(temp)):
                post_t = (temp[k])
                p=next(item for item in posts if item["t"] == post_t)
                text =  text + " " + str(p['original_text'])
            ## Processing the text
            emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
            emoji_pattern.sub(r'', text)   
            text = re.sub(r"[a-zA-Z_0-9]", " ", text)
            not_text=["、","（",".","）",".....","?","“",".","《","》","！","'","”","`","•","з","☆","^","」"," ∠","，","[","]","(",")","：",";","；","/","！","？","-","@",":","。",",","·","…","→","_","=","】","【","∀","~","～","*"]
            analyzer = ChineseAnalyzer() #Chinese words parsing
            result = analyzer.parse(text)
            result = result.tokens() 
            final_text=""
            for elem in result:
                if elem !=' ' and elem not in not_text:
                    final_text= final_text + " " + elem    
            pre_event.append(final_text)
        
        #Caculate Tfidf
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        try:
            tfsc = vectorizer.fit_transform(pre_event)
            tfidf = transformer.fit_transform(tfsc)
            RNN_data.append((tfidf.toarray()).tolist())
        except:
            RNN_data.append([0]*len(pre_event))
    
    
RNN_data_test = []
RNN_data_train = []
RNN_data_val = []

## /!\ Very slow save txt files with pickle
# features_extract(event_related_posts_test,event_ids_test, time_series_test,RNN_data_test) 
# with open("RNN_data_test.txt", "wb") as fp:
#         pickle.dump(RNN_data_test, fp)
# features_extract(event_related_posts_val,event_ids_val, time_series_val,RNN_data_val) 
# with open("RNN_data_val.txt", "wb") as fp:
#         pickle.dump(RNN_data_val, fp) 
# features_extract(event_related_posts_train,event_ids_train, time_series_train,RNN_data_train)    
# with open("RNN_data_train.txt", "wb") as fp:
#         pickle.dump(RNN_data_train, fp)


  
# Load pickled text files      
with open("RNN_data_test.txt", "rb") as fp:   # Unpickling
          RNN_data_test=(pickle.load(fp))        
with open("RNN_data_val.txt", "rb") as fp:   # Unpickling
          RNN_data_val=(pickle.load(fp))
with open("RNN_data_train.txt", "rb") as fp:   # Unpickling
          RNN_data_train=(pickle.load(fp))        

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
k = 1500 #the number of tf.idf values sorted in descending order we will keep for each interval
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
    
###################################################################################
#                                                                                 #
#       RNN                                                                       #
#                                                                                 #
################################################################################### 
embeddin_size=100

RNN_data_train=np.array(new_rnn_train)
RNN_data_test=np.array(new_rnn_test)
RNN_data_val=np.array(new_rnn_val)

# define the based sequential model
model = Sequential()
# RNN layer
model.add(Dense(embeddin_size, input_shape=(N,k)))
model.add(Dropout(0.3))
model.add(LSTM(N,input_shape = (N, embeddin_size),return_sequences=False))
# Output layer for classification
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(
    loss='categorical_crossentropy',
    #optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.005, initial_accumulator_value=0.1, epsilon=1e-07),
    #optimizer= tf.keras.optimizers.Adam(lr=1e-2, decay=1e-5),
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
    regularizer=tf.keras.regularizers.l2(l=0.01),
    metrics=['accuracy'],
)

# Train and test the model
model.fit(RNN_data_train,
          labels_train_onehot,
          epochs=50,
          batch_size=32,
          validation_data=(RNN_data_test, labels_test_onehot))

# Evaluate the model
pred = model.predict(RNN_data_val)
y_pred = np.argmax(pred, axis=1)
print("Accuracy={:.2f}".format(np.mean(pred ==labels_val_onehot )))




    
        

