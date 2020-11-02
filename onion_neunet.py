import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide TensorFlow warnings
import timeit # to calculate run time
start = timeit.default_timer()

from math import log
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from keras import optimizers
from keras import metrics
from keras.callbacks.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

# parameters
hid_neurons = 100  # neurons in hidden layer
epochs = 100
batch = 1000
lr = 1
momentum = 0.2

# import csv
dataset = pd.read_csv('dataset.csv')

# split dataset
train_dataset = dataset.iloc[:int(0.75*len(dataset))]
test_dataset = dataset.loc[int(0.75*len(dataset)):]

# create early stopping callback
es = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=True)

# create network architecture one level at a time (sequentially)
net = Sequential()
net.add(Dense(hid_neurons, input_dim=len(dataset.columns)-1, activation='relu'))  # hidden layer
net.add(Dense(1, input_dim=hid_neurons, activation='relu'))  # output layer

#parameters of gradient decent
sgd = optimizers.SGD(lr=lr, momentum=momentum)  # learning rate, momentum
net.compile(loss='mean_absolute_error', optimizer=sgd, metrics=[metrics.Precision(),metrics.Recall()])

# train network
x_train = train_dataset.iloc[:,:-1] # train inputs
y_train = train_dataset.iloc[:,-1] # train outputs
x_test = test_dataset.iloc[:,:-1] # test inputs
y_test = test_dataset.iloc[:,-1] # true test outputs
history = net.fit(x_train, y_train, batch_size=batch, epochs=epochs, callbacks=[es], validation_split=0.2, verbose=1)

# test network
scores = net.evaluate(x_test, y_test, batch_size=batch, verbose=0)

# print results
precision = scores[1]
recall = scores[2]
f1 = 2*precision*recall/(precision+recall)
print(f'F1 = {f1:.4f}')
print(f'Precision = {precision:.4f}')
print(f'Recall = {recall:.4f}')

stop = timeit.default_timer()
print(f'Runtime: {int((stop - start)/60)} minutes')

import winsound
duration = 500  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)