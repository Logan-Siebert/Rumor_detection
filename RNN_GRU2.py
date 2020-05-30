import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
###################################################################################
#                                                                                 #
#       RNN                                                                       #
#                                                                                 #
###################################################################################

#Load the preprocessed data

with open('RNN_data_train.npy', 'rb') as f:
    RNN_data_train = np.load(f)

with open('RNN_data_test.npy', 'rb') as f:
    RNN_data_test = np.load(f)

with open('RNN_data_val.npy', 'rb') as f:
    RNN_data_val = np.load(f)

with open('labels_train_onehot.npy', 'rb') as f:
    labels_train_onehot = np.load(f)

with open('labels_test_onehot.npy', 'rb') as f:
    labels_test_onehot = np.load(f)

with open('labels_val_onehot.npy', 'rb') as f:
    labels_val_onehot = np.load(f)


N = RNN_data_train.shape[1]
k = RNN_data_train.shape[2]

embeddin_size=100

# define the based sequential model
model = Sequential()
# RNN layers
lamb = 0.005
model.add(Dense(embeddin_size, input_shape=(N,k), kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
model.add(GRU(N,input_shape = (N, embeddin_size),return_sequences=True, kernel_regularizer = tf.keras.regularizers.l2(lamb)))
#model.add(Dropout(0.1)) #Dropout
model.add(GRU(N,input_shape = (N, embeddin_size),return_sequences=False,kernel_regularizer = tf.keras.regularizers.l2(lamb)))
#model.add(Dropout(0.1)) #Dropout
# Output layer for classification
model.add(Dense(2, activation='softmax'))
model.summary()

tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

model.compile(
    loss='categorical_crossentropy',
    #optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.005, initial_accumulator_value=0.1, epsilon=1e-07),
    optimizer= tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5),
    #optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
    #regularizer=tf.keras.regularizers.l2(l=0.1),
    metrics=['accuracy'],
)

# Train and test the model
model.fit(RNN_data_train,
          labels_train_onehot,
          epochs=100,
          batch_size=32,
          validation_data=(RNN_data_test, labels_test_onehot))

# Evaluate the model
pred = model.predict(RNN_data_val)
y_pred = np.argmax(pred, axis=1)
lab= np.argmax(labels_val_onehot, axis=1)
print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))
