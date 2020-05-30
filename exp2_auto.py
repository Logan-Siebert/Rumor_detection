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

learningRate = 1e-3
while learningRate < 8e-3 :

    N = RNN_data_train.shape[1]
    k = RNN_data_train.shape[2]

    embeddin_size = 100
    amount_runs = 10
    count = 0
    embeddin_size = 100

    #Accuracies
    test_accuracies = []
    train_accuracies = []
    val_accuracies = []

    dropout = 0.3
    lamb = 0.001

    maxEpochs = 20
    allEpochs = []
    arch = 0   # 0 --> simpleRNN, 1 --> LSTM, 2--> GRU
    opti = 'Adam'


    # Architecture SimpleRNN -------------------------------------------------------

    if arch == 0 :
        while count < amount_runs :
            # define the based sequential model
            model = Sequential()
            # RNN layers
            model.add(Dense(embeddin_size, input_shape=(N,k),
                            kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
            model.add(SimpleRNN(N,
                                input_shape = (N, embeddin_size),
                                return_sequences=False,
                                kernel_regularizer = tf.keras.regularizers.l2(lamb)))
            model.add(Dropout(dropout)) #Dropout

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
                      batch_size=32,
                      validation_data=(RNN_data_test, labels_test_onehot))

            # Evaluate the model
            pred = model.predict(RNN_data_val)
            y_pred = np.argmax(pred, axis=1)
            lab= np.argmax(labels_val_onehot, axis=1)

            #Recording accuracies
            test_accuracies.append(np.mean(y_pred ==lab))
            train_accuracies.append(np.mean(model_history.history["accuracy"]))
            val_accuracies.append(np.mean(model_history.history["val_accuracy"]))

            print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))
            count += 1

    # Architecture LSTM -------------------------------------------------------
    if arch == 1 :
        while count < amount_runs :
            # define the based sequential model
            model = Sequential()
            # RNN layers
            model.add(Dense(embeddin_size, input_shape=(N,k),
                            kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
            model.add(LSTM(N,
                           input_shape = (N, embeddin_size),
                           return_sequences=False,
                           kernel_regularizer = tf.keras.regularizers.l2(lamb)))
            model.add(Dropout(dropout)) #Dropout

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
            train_accuracies.append(np.mean(model_history.history["accuracy"]))
            val_accuracies.append(np.mean(model_history.history["val_accuracy"]))

            print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))
            count += 1

    # Architecure GRU --------------------------------------------------------------


    if arch == 2 :
        while count < amount_runs :
            # define the based sequential model

            model.add(Dense(embeddin_size, input_shape=(N,k), kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
            model.add(GRU(N,input_shape = (N, embeddin_size),return_sequences=True, kernel_regularizer = tf.keras.regularizers.l2(lamb)))
            #model.add(Dropout(0.1)) #Dropout
            model.add(GRU(N,input_shape = (N, embeddin_size),return_sequences=False,kernel_regularizer = tf.keras.regularizers.l2(lamb)))
            #model.add(Dropout(0.1)) #Dropout
            # Output layer for classification
            model.add(Dense(2, activation='softmax'))
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
            train_accuracies.append(np.mean(model_history.history["accuracy"]))
            val_accuracies.append(np.mean(model_history.history["val_accuracy"]))
            print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))
            count += 1

    E_test = 0
    E_train = 0
    E_val = 0

    S_test = 0
    S_train = 0

    #Mean over testing accuracy
    for i in range(len(test_accuracies)) :
        E_test += test_accuracies[i]
    E_test = E_test/len(test_accuracies)

    #Mean over testing accuracy
    for i in range(len(train_accuracies)) :
        E_train += train_accuracies[i]
    E_train = E_train/len(train_accuracies)

    #Mean over val accuracy
    for i in range(len(val_accuracies)) :
        E_val += val_accuracies[i]
    E_val = E_val/len(val_accuracies)

    #\sigma^2 over experiments - test
    for i in range(len(test_accuracies)) :
        S_test += (test_accuracies[i] - E_test)**2
    S_test = E_test/(len(test_accuracies))

    #\sigma^2 over experiments - train
    for i in range(len(test_accuracies)) :
        S_train += (test_accuracies[i] - E_train)**2
    S_train = E_train/(len(test_accuracies))


    print("E(Accuracy)" + str(E_test) + "Amount experiments : " + str(amount_runs))

    # Saving results of experiment for Panda

    """
    opt arch MaxEpochs E(test_accuracy) S(test_accuracy) E(train_accuracy) S(train_accuracy) E(val_accuracy) nb(experiments) lr embedding K reg dropout
    """
    archi =''

    if(arch == 0) :
        archi = 'SimpleRNN'
    if(arch == 1) :
        archi = 'LSTM'
    if(arch == 2) :
        archi = 'GRU'

    with open('expData.csv', 'a') as file:
        line = opti + ' ' + archi + ' ' + str(maxEpochs) + ' ' + str(round(E_test, 5)) + ' ' + str(round(S_test, 5)) + ' ' + str(round(E_train, 5)) + ' ' + str(round(S_train, 5)) + ' ' + str(round(E_val, 5)) + ' ' + str(count) + ' ' + str(learningRate) + ' ' + str(embeddin_size) + ' ' + str(k) + ' ' + str(lamb) + ' ' + str(dropout) + '\n'
        file.write(line)
        learningRate += 1e-3
