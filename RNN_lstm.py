"""
File description : RNN implementation using Tesorflow2 for classification problem
                   Architecture : sequencial RNN
                                  LSTM - two embedding layers

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

#Visualization imports
import analysis.eventEvolution as ev
import analysis.plotAccuracy as pltA

n_ev = 4664 # number of evenements


N=20; # RNN reference length N

###################################################################################
#                                                                                 #
#       Unpacking preprocessed data                                               #
#                                                                                 #
###################################################################################


#Data
with open("Data/Preprocessed/train.txt", "rb") as fp:   # Unpickling
          rnn_train=(pickle.load(fp))
with open("Data/Preprocessed/test.txt", "rb") as fp:   # Unpickling
          rnn_test=(pickle.load(fp))

#Label
with open("Data/Preprocessed/label_train.txt", "rb") as fp:   # Unpickling
          labels_train=(pickle.load(fp))
with open("Data/Preprocessed/label_test.txt", "rb") as fp:   # Unpickling
          labels_test=(pickle.load(fp))


#Checking the Data
#Just checking what the max number of tf.idf values (maxK) inside any interval in the data is
maxK = 0
for event in rnn_train:
    maxK = max(len(max(event,key=len)),maxK)
print("Largest K value in all intervals = " + str(maxK))
maxNrIntervals = max( len(max(rnn_train,key=len)), len(max(rnn_test,key=len)))
print("Largest # intervals in a single event = " + str(maxNrIntervals))


import sys
#Tensorflow RNN
tf.device('/gpu:1') #My best gpu is gpu:1, change to gpu:0 if you only have 1 gpu
# CUDA_VISIBLE_DEVICES=1
TF_CPP_MIN_LOG_LEVEL=2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, SimpleRNN

model = Sequential()

# Building the model ------------------------------------------------------------

# Embedding layer - Reducing problem dimensionnability

model.add(SimpleRNN(maxNrIntervals, activation='tanh',use_bias=True, kernel_initializer='uniform',
                   recurrent_initializer='orthogonal', kernel_regularizer=tf.keras.regularizers.l2(l=1),
                    bias_initializer='zeros',dropout=0.0, recurrent_dropout=0.0,
                   return_sequences=False))


model.add(Dense(2,activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=1)))

opt = tf.keras.optimizers.Adagrad(lr=0.1, initial_accumulator_value=0.9, epsilon=1e-07)
#opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)


# Model compilation -------------------------------------------------------------

model.compile(
    loss='categorical_crossentropy',
    #this should be mse between the probability distributions of the prediction and ground truth + L2-regularization penalty
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(rnn_train,
          labels_train,
          epochs=10,
          batch_size=10,
          validation_data=(rnn_test, labels_test))

model.summary()

###################################################################################
#                                                                                 #
#       Analysis                                                                  #
#                                                                                 #
###################################################################################
