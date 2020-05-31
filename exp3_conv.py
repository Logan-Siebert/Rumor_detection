"""
File description : Running experiments for varying parameters to find optimal
                   configuration

                   K : 500 -> 2500 -> 5000
                   lr : 1e-3 -> 4e-3 -> 7e-3 -> 10e-3
                   reg_lambda -> 0.00001 -> 0.0001 -> 0.001
"""

import numpy as np
import os
import math as m
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU

###################################################################################
#                                                                                 #
#       RNN                                                                       #
#                                                                                 #
###################################################################################

#Load the preprocessed data
with open('labels_train_onehot.npy', 'rb') as f:
    labels_train_onehot = np.load(f)

with open('labels_test_onehot.npy', 'rb') as f:
    labels_test_onehot = np.load(f)

with open('labels_val_onehot.npy', 'rb') as f:
    labels_val_onehot = np.load(f)

def saveVector(list) :
    """Writing the accuracy vector list
    """
    with open('conv.csv', 'a') as file:
        line = ''
        for i in range(len(list)) :
            line += str(list[i]) +', '
        file.write(line)

def saveVector2(list) :
    """Writing the accuracy vector list
    """
    with open('conv2.csv', 'a') as file:
        line = ''
        for i in range(len(list)) :
            line += str(list[i]) +', '
        file.write(line)


# Fixed experiment parameters
amount_runs = 10   # Runs per experiments
maxEpochs = 100
allEpochs = []
arch = 1  # 0 --> simpleRNN, 1 --> LSTM, 2--> GRU
opti = 'Adam'
dropout = 0.2      #Dropout goodpractice ~20%
embeddin_size = 100

allE = []
allLoss = []
# Has to be adapted to the way you handle K
k = 2500

#loading data, files must exist (should implement exception handling)
fileNameTrain = 'RNN_data_train' + str(k) + '.npy'
fileNameTest = 'RNN_data_test' + str(k) + '.npy'
fileNameVal = 'RNN_data_val' + str(k) + '.npy'


if os.path.exists(fileNameTrain):
   with open(fileNameTrain, 'rb') as f:
       try:
           RNN_data_train = np.load(f)
       except :
           print("Error : RNN_data_train for k : " + str(k) + "doesn't exist")

if os.path.exists(fileNameTest):
   with open(fileNameTest, 'rb') as f:
       try:
           RNN_data_test = np.load(f)
       except :
           print("Error : RNN_data_test for k : " + str(k) + "doesn't exist")

if os.path.exists(fileNameVal):
   with open(fileNameVal, 'rb') as f:
       try:
           RNN_data_val = np.load(f)
       except :
           print("Error : RNN_data_valfor k : " + str(k) + "doesn't exist")

#Reshaping data
N = RNN_data_train.shape[1]
k = RNN_data_train.shape[2]

learningRate = 0.001


#Experiments through regularization \lambda values
lamb = 0.01
count = 0

#Accuracies
test_accuracies = []
train_accuracies = []
val_accuracies = []
val_loss= []

# Architecture SimpleRNN -------------------------------------------------------
if arch == 0 :
    # define the based sequential model
    model = Sequential()
    # RNN layers
    model.add(Dense(embeddin_size, input_shape=(N,k),
                    kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
    model.add(SimpleRNN(N,
                        input_shape = (N, embeddin_size),
                        return_sequences=False,
                        kernel_regularizer = tf.keras.regularizers.l2(lamb)))
    # model.add(Dropout(dropout)) #Dropout

    # Output layer for classification
    model.add(Dense(2, activation='softmax',
                    kernel_regularizer = tf.keras.regularizers.l2(lamb)))
    model.summary()

    # tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    #     baseline=None, restore_best_weights=False
    # )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=learningRate, initial_accumulator_value=0.1, epsilon=1e-07),
        #optimizer= tf.keras.optimizers.Adam(lr=learningRate, decay=1e-5),
        #optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
        #regularizer=tf.keras.regularizers.l2(l=lamb),
        metrics=['accuracy'],
    )

    # Train and test the model
    model_history = model.fit(RNN_data_train,
              labels_train_onehot,
              epochs=maxEpochs,
              batch_size=64,
              validation_data=(RNN_data_test, labels_test_onehot))

    # Evaluate the model
    pred = model.predict(RNN_data_val)
    y_pred = np.argmax(pred, axis=1)
    lab= np.argmax(labels_val_onehot, axis=1)

    #Recording accuracies
    saveVector(model_history.history["val_accuracy"])
    saveVector2(model_history.history["train_accuracy"])


    print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))

# Architecture LSTM -------------------------------------------------------
if arch == 1 :
    # define the based sequential model
    model = Sequential()
    # RNN layers
    model.add(Dense(embeddin_size, input_shape=(N,k),
                    kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
    model.add(LSTM(N,
                   input_shape = (N, embeddin_size),
                   return_sequences=False,
                   kernel_regularizer = tf.keras.regularizers.l2(lamb)))
    # model.add(Dropout(dropout)) #Dropout

    # Output layer for classification
    model.add(Dense(2, activation='softmax',
                    kernel_regularizer = tf.keras.regularizers.l2(lamb)))
    model.summary()


    # tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    #     baseline=None, restore_best_weights=False
    # )

    model.compile(
        loss='categorical_crossentropy',
        # optimizer=tf.keras.optimizers.Adagrad(learning_rate=learningRate, initial_accumulator_value=0.1, epsilon=1e-07),
        optimizer= tf.keras.optimizers.Adam(lr=learningRate, decay=1e-5),
        #optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
        # regularizer=tf.keras.regularizers.l2(l=lamb),
        metrics=['accuracy'],
    )

    # Train and test the model
    model_history = model.fit(RNN_data_train,
              labels_train_onehot,
              epochs=maxEpochs,
              batch_size=32,
              validation_data=(RNN_data_test, labels_test_onehot))

    # Evaluate the model
    pred = model.predict(RNN_data_val)
    y_pred = np.argmax(pred, axis=1)
    lab= np.argmax(labels_val_onehot, axis=1)

    #Recording accuracies
    test_accuracies.append(np.mean(y_pred ==lab))
    #Recording accuracies
    saveVector(model_history.history["val_accuracy"])
    saveVector2(model_history.history["accuracy"])


    print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))

# Architecure GRU --------------------------------------------------------------


if arch == 2 :
    # define the based sequential model
    model = Sequential()

    model.add(Dense(embeddin_size, input_shape=(N,k), kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
    model.add(GRU(N,input_shape = (N, embeddin_size),return_sequences=True, kernel_regularizer = tf.keras.regularizers.l2(lamb)))
    #model.add(Dropout(dropout)) #Dropout
    model.add(GRU(N,input_shape = (N, embeddin_size),return_sequences=False,kernel_regularizer = tf.keras.regularizers.l2(lamb)))
    #model.add(Dropout(dropout)) #Dropout
    # Output layer for classification
    model.add(Dense(2, activation='softmax',kernel_regularizer = tf.keras.regularizers.l2(lamb)))
    model.summary()
    # tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    #     baseline=None, restore_best_weights=False
    # )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=learningRate, initial_accumulator_value=0.1, epsilon=1e-07),
        #optimizer= tf.keras.optimizers.Adam(lr=learningRate, decay=1e-5),
        #optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
        # regularizer=tf.keras.regularizers.l2(l=lamb),
        metrics=['accuracy'],
    )

    # Train and test the model
    model_history = model.fit(RNN_data_train,
              labels_train_onehot,
              epochs=maxEpochs,
              batch_size=32,
              validation_data=(RNN_data_test, labels_test_onehot))

    # Evaluate the model
    pred = model.predict(RNN_data_val)
    y_pred = np.argmax(pred, axis=1)
    lab= np.argmax(labels_val_onehot, axis=1)

    #Recording accuracies
    test_accuracies.append(np.mean(y_pred ==lab))
    #Recording accuracies
    saveVector(model_history.history["val_accuracy"])
    saveVector2(model_history.history["train_accuracy"])


    print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))

#computing std and E
