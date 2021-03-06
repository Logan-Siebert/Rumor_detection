"""
File description : ANN model classifyer for any properly defined classification
                   problem.

                   N = input vector size
                   2 = Output size
"""

#Base imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import sys

# Variables
N = 50
#opti = "Adagrad"
PATHVEC = 'Data/TwitterData/Vectors/'

# Loading data build in doc2vec script (raw)
xTrain = np.load(PATHVEC + 'xTrain.npy')
yTrain = np.load(PATHVEC + 'yTrain.npy')
xTest = np.load(PATHVEC + 'xTest.npy')
yTest = np.load(PATHVEC + 'yTest.npy')

# Converting y's into categroricals (two-output vectors)
yTrain = tf.keras.utils.to_categorical(yTrain)
yTest = tf.keras.utils.to_categorical(yTest)


###################################################################################
#                                                                                 #
#       Preparing data                                                            #
#                                                                                 #
###################################################################################


#Building model ----------------------------------------------------------------

model = tf.keras.Sequential()
tf.device('/gpu:1') #My best gpu is gpu:1, change to gpu:0 if you only have 1 gpu

#Input shape
model.add(tf.keras.layers.BatchNormalization(batch_input_shape = (None, N)))
model.add(tf.keras.layers.Dense(N + 50))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))

#Output
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Activation('softmax'))

#Compile
model.compile(optimizer = tf.keras.optimizers.Adagrad(),
              loss = tf.keras.losses.categorical_crossentropy,
              metrics = [tf.keras.metrics.categorical_crossentropy,
              tf.keras.metrics.categorical_accuracy])

model.fit(x = xTrain,
          y = yTrain,
          validation_data = (xTest, yTest),
          batch_size = 64,
          epochs = 150,
          verbose = 1)

model.summary()
