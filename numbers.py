import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D, AvgPool2D
from keras.layers import Flatten, Activation
import keras.backend as K

# Import MNIST dataset
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data('/tmp/mnist.npz')

print('Array of greyscale values')
print(X_train[0])
print('Each image is 28x28')

# Reshape to vectors
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)
# Turn into 0-1 percentages
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
# Shape now = (60000, 784)

# Change outputs into categories
# y_train[0] = 5
# y_train_cat[0] = [0,0,0,0,0,1,0,0,0,0]
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# Build network
K.clear_session()
model = Sequential()
model.add(Dense(512, input_dim=28*28, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Train model
fc = model.fit(X_train, y_train_cat, batch_size=128, epochs=3, verbose=1, validation_split=0.3)

# Test accuracy
test_accuracy = model.evaluate(X_test, y_test_cat)[1]
print('Fully Connected NN test accuracy = ', test_accuracy)

''' Convolutional Neural Net '''

# Reshape data
# shape now = (60000, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

K.clear_session() # clears CPU

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Flatten()) # Flatten to feed into last layers
# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print('\n\nConvolutional Neural Net')
print('\nSummary of layers -- notice how parameter numbers, output sizes work')
model.summary()

cnn = model.fit(X_train, y_train_cat, batch_size=128,
          epochs=3, verbose=1, validation_split=0.3)

# Test accuracy
test_accuracy = model.evaluate(X_test, y_test_cat)[1]
print('CNN test accuracy = ', test_accuracy)
