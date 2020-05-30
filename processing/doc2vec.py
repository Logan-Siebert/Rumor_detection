"""
File description : Extracting textual data from the labeled events
                   Then Extracting features with doc2vec method

                   Formatting : Twitter.txt
                                900 labled events
                                each line -> one event with ids of relevant posts
                                (event_id, label, post_ids)

                   Content of the posts in .json format (event_id.json)
"""

#System
from __future__ import print_function
import os

import json as js
import numpy as np
import random
import pickle
from collections import OrderedDict

#Text processing libs
import string
import gensim as gen  #doc2vec/word2vec lib
from gensim.models import Doc2Vec as d2v
import re             # Cleaning text

"""
    Data structured as tweets ids as keys, labels in Twitter.txt, associated
    to all related ids.
"""

PTH = '../Data/Twitter.txt'
FILES = '../Data/TwitterData/tweets/'
PATHMODEL ='../Data/TwitterData/Models/'
PATHVEC = '../Data/TwitterData/Vectors/'

# General parameters

ANN_vector_size = 50

###################################################################################
#                                                                                 #
#       Word Processing                                                           #
#                                                                                 #
###################################################################################

# Cleaning up tweets -----------------------------------------------------------

def cleanText(text):
    """Takes a string, returns an array of words, having removed all ponctuation
    and special symbols.

    Use of the regex model, keeping only a - Z ASCII values, converting
    to lower cases.

    We need to work with whole strings instead of splitting for the gensim lib
    """

    # Converting to lower cases and removing contractions, removing urls, hashtags, mentions

    #DEBUG --> has to be improved
    # [] Removal of one word text
    # [] Some numbers are passing through
    # [] Some twitter handles are passing through

    if isinstance(text, str) :

        # Removing URLS
        text = re.sub(r'http\S+|https\S+|www.\S+', '', text)
        text = re.sub(r'pic.twitter\S+', '', text)

        #Removing twitter handles and hashtags
        text = re.sub(r'@[A-Za-z0-9]+','',text)
        text = re.sub(r'#[A-Za-z0-9]+','',text)

        #Removing numbers and ponctuation
        text  = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        text = re.sub(r"\\xa0\S+", "", text)

        # Removing automatic words, non-ascii and one-word word, setting to lowercase
        text = ' '.join( [w for w in text.split() if len(w)>1] )
        text = re.sub(r"[^a-zA-Z]+", " ", text)
        text = re.sub(r"via", '', text)
        text = text.lower()
        #print(text)

        return text
    else :
        return None

# Extracting all tweets structured in a dictionnary ----------------------------

"""
Using the dic structure allowes for easier splitting and moving data for shuffling
"""
# Building dictionnary

dic = {}
allKeys = []   #might not be useful


"""
eid : ID OF EVENT
label : label

--> List of relevant twitter ids end --> \n
"""

print("Loading twitter data ***.json --> labelled by ids Twitter.txt")
with open(PTH) as rawTwitterFile :

    # Reading the file line per line
    for line in rawTwitterFile :
        splitline = line.rstrip()
        splitline = splitline.split(' ')   # Space splitting

        eid = splitline[0].split(':')[1]
        label = int(splitline[1].split(':')[1])

        # Adding all tweets ids
        tweetsIds = []
        for i in range(2, len(splitline)) :
            tweetsIds.append(splitline[i])
            dic[eid] = (tweetsIds, label)
        allKeys.append(eid)
        #print("Made structured event - label: ", eid, label)

# Splitting data, shuffling - making training and testing arrays ---------------

sortedEvents = sorted(dic.items(), key=lambda eID: eID[1])    #classification on the keys
random.shuffle(sortedEvents)

trainingKeyList = sortedEvents[0:int(0.80*len(sortedEvents))]
testingKeyList = sortedEvents[int(0.80*len(sortedEvents)):]

print("Training : " + str(len(trainingKeyList)) + "| Testing : " + str(len(testingKeyList)) + "| Total Event :" + str(len(allKeys)))
# Recreating the split dictionnary
dictTrain = OrderedDict(trainingKeyList)
dictTest = OrderedDict(testingKeyList)

###################################################################################
#                                                                                 #
#       doc2vec                                                                   #
#                                                                                 #
###################################################################################

# Creating doc2vec arrays ------------------------------------------------------

trainArray = []
testArray = []
count = 0
tempDictionnary = {}

countTrainingData = 0
#Training Array
for key, inf in dictTrain.items() :

    filepath = FILES + key + ".json"

    # Trying to load the json, if the file skips of the file doesn't exist
    if os.path.isfile(filepath):
        tempVector = []

        with open(filepath, "r", encoding='utf-8') as dataFile :
            tempDictionnary = js.loads(dataFile.read())

        for twKey, twText in tempDictionnary.items() :

            tweetContent = cleanText(twText[1])   #0 index is the date which we don't use, could be extracted as feature later
            if tweetContent is not None :
                tempVector.extend(tweetContent)
        countTrainingData +=1
    trainArray.append(gen.models.doc2vec.LabeledSentence(tempVector, key))
    #print(trainArray)
    #print(trainArray[0][0][7])

#Testing Array
countTestingData = 0
for key, inf in dictTest.items() :

    filepath = FILES + key + ".json"

    # Trying to load the json, if the file skips of the file doesn't exist
    if os.path.isfile(filepath):
        tempVector = []

        with open(filepath, "r", encoding='utf-8') as dataFile :
            tempDictionnary = js.loads(dataFile.read())

        for twKey, twText in tempDictionnary.items() :

            tweetContent = cleanText(twText[1])   #0 index is the date which we don't use, could be extracted as feature later
            if tweetContent is not None :
                tempVector.extend(tweetContent)
        countTestingData += 1
    testArray.append(gen.models.doc2vec.LabeledSentence(tempVector, key))


###################################################################################
#                                                                                 #
#       word2vec importance training                                              #
#                                                                                 #
###################################################################################


print("Training started ------- gensim ANN doc2vec model")
maxEpoch = 9
alpha = 0.025

#Defining model
model = gen.models.Doc2Vec(vector_size = ANN_vector_size,
                alpha = alpha,
                minAlpha = 0.00025,
                minCount = 1,
                dm = 1)
model.build_vocab(trainArray)

#Training doc2vec
for epoch in range(maxEpoch):
    print('Epoch :  {0}'.format(epoch), end='\r')
    model.train(trainArray,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

pathModel = PATHMODEL + 'd2v.model'
model.save(pathModel)


# Creating the ANN inputs ------------------------------------------------------

"""
Creating input ANN vector : (amount of documents x ANN_vector_size)
    For training and testing

Creating output ANN vector : (amount of documents x 1)
    For training and testing
"""
xTrain = np.zeros((len(trainArray), ANN_vector_size))      # Input values
yTrain = np.zeros((len(trainArray)))      # Associated labels -- > 1D array

xTest = np.zeros((len(testArray), ANN_vector_size))
yTest = np.zeros((len(testArray)))

# Infering associated vector from model and labelling vectors ------------------

# Training
it = 0
for event in trainArray :
    inputVector = model.infer_vector(event.words)
    xTrain[it, ] = inputVector
    #print(xTrain[it, ])
    yTrain[it] = dic[event.tags][1]
    #print(yTrain[it])
    it+=1

# Testing
it = 0
for event in testArray :
    inputVector = model.infer_vector(event.words)
    xTest[it, ] = inputVector
    #print(xTrain[it, ])
    yTest[it] = dic[event.tags][1]
    #print(yTest[it])
    it+=1

# Normalizing ------------------------------------------------------------------
# normTrain = np.linalg.norm(xTrain)
# normTest = np.linalg.norm(xTest)
#
# xTrain = xTrain/normTrain
# xTest = xTest/normTest


# Serializing ------------------------------------------------------------------
x_Path_training = PATHVEC + 'xTrain'
y_Path_training = PATHVEC + 'yTrain'
x_Path_testing = PATHVEC + 'xTest'
y_Path_testing = PATHVEC + 'yTest'

np.save(x_Path_training, xTrain)
np.save(y_Path_training, yTrain)
np.save(x_Path_testing, xTest)
np.save(y_Path_testing, yTest)

#Pickling can't be done on numpy arrays, just use the save function
# with open(x_Path_training, "wb") as fp:
#       pickle.dump(xTrain, fp)
# with open(y_Path_training, "wb") as fp:
#       pickle.dump(yTrain, fp)
# with open(xTest, "wb") as fp:
#       pickle.dump(xTest, fp)
# with open(y_Path_testing, "wb") as fp:
#       pickle.dump(yTest, fp)



# Summary
print("Model build --\n")
print("========================================================================")
print("    Training doc2vec : " + str(countTrainingData) + " ")
print("    Testing doc2vec  : " + str(countTestingData)+ " ")
print("\n")
print("    Missing files : " + str(len(allKeys) - (countTrainingData + countTestingData)))
print("------------------------------------------------------------------------")
print("    Model : Epochs : " + str(maxEpoch) + "     Vect size : " + str(ANN_vector_size) + "     gensim LR : " + str(alpha))
print("    Saved model at " + PATHMODEL)
print("------------------------------------------------------------------------")
print("    Saved infered ANN vectors : ")
print("      xTraining at : " + x_Path_training)
print("      yTraining at : " + y_Path_training)
print("      xTesting at :  " + x_Path_testing)
print("      yTesting at :  " + y_Path_testing)
print("========================================================================")
