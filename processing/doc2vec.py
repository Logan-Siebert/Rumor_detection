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
from collections import OrderedDict

#Text processing libs
import gensim as gen  #doc2vec/word2vec lib
from gensim.models import Doc2Vec as d2v

import re             # Cleaning text

"""
    Data structured as tweets ids as keys, labels in Twitter.txt, associated
    to all related ids.
"""

PTH = '../Data/Twitter.txt'
FILES = '../Data/TwitterData/tweets/'


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
    if isinstance(text, str) :

        to_match = ['http\S+',
            '@\S+',
            '[0-9]+']

        text = re.sub('|'.join(to_match), '', text)
        text = re.sub(r"\\xa0\S+", "", text)
        text = re.sub(r"[^a-zA-Z]+", " ", text)
        text = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", "", text)
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

        text.lower()
        allWords = re.split(r'\W+', text)    #Regex word model

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

#print(dictTrain.items())

# Creating doc2vec arrays ------------------------------------------------------

trainArray = []
testArray = []
count = 0
tempDictionnary = {}

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
                tempVector.append(tweetContent)
    trainArray.append(gen.models.doc2vec.LabeledSentence(tempVector, key))
    print(trainArray)

#Testing Array
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
                tempVector.append(tweetContent)
    testArray.append(gen.models.doc2vec.LabeledSentence(tempVector, key))
    print(trainArray)


# Serializing ------------------------------------------------------------------


###################################################################################
#                                                                                 #
#       word2vec importance training                                              #
#                                                                                 #
###################################################################################
