#!/usr/bin/python -u
# Adapted from https://github.com/mashgin/basic_deep_learning_keras

import numpy as np
import os, sys, random
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model


np.random.seed(1337)

classes = 2
rows, cols, channels = 100, 100, 1
train_samples = 500
test_samples = int(train_samples * 0.2)

# Generate a single random shape (circle or square)
def generate_shape():
	# Choose a random shape
	cls = random.randint(0, classes - 1)
	img = np.zeros((rows, cols, channels), dtype='uint8')

	# Circle of random size
	if cls == 0:
		cv2.circle(img, (int(cols/2), int(rows/2)),
			random.randint(10, int(cols/2 - 10)), (255), -1)
	# Square of random size
	elif cls == 1:
		side = random.randint(10, cols/2 - 10)
		cv2.rectangle(img, (int(cols/2 - side), int(cols/2 - side)),
					(int(cols/2 + side), int(cols/2 + side)), (255), -1)

	chans = cv2.split(img)
	out_image = np.array(chans, dtype='float32') / 255.

	return cls, out_image

# Return an array of multiple samples (labels, images)
def generate_samples(batch_size):
	batch_labels, batch_images = [], []

	for i in range(0, batch_size):
		cls, image = generate_shape()
		batch_images.append(image)
		batch_labels.append(cls)

	return np.array(batch_labels).reshape(batch_size, 1), np.array(batch_images)

# Model
def GetModel():
	# Define Neural Net layers
	model = Sequential()
	model.add(Conv2D(16, (5, 5), input_shape=(rows, cols, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	# model.add(Conv2D(32, (5, 5)))
	# model.add(Activation('relu'))
	# model.add(MaxPooling2D(pool_size=(2,2)))
	# model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def train():
	'''
		Train the neural net and test the accuracy every couple iterations
	'''
	model = GetModel()

	# Create & reshape data
	labels, imgs = generate_samples(train_samples)
	imgs_train = imgs.reshape(train_samples, rows, cols, 1)
	labels_train = np_utils.to_categorical(labels, classes)

	# Train model
	print('')
	model.fit(imgs_train, labels_train, batch_size=100,
		  epochs=4, verbose=1, validation_split=0.2)

	# Test accuracy
	labels, imgs = generate_samples(test_samples)
	imgs_test = imgs.reshape(test_samples, rows, cols, 1)
	labels_test = np_utils.to_categorical(labels, classes)

	print('\nEvaluating on test data...')
	test_accuracy = model.evaluate(imgs_test, labels_test)[1]
	print('Test accuracy = ', test_accuracy)

	# Save to file for later use
	model.save('./weights.h5')


def classify():
	'''
		Load the trained network
	'''
	model = load_model('./weights.h5')

	'''
		For fancy output
	'''
	vis_dictionary = {0:"circle",1:"square"}

	'''
		Get an image, let the network  predict what it is,
		output the image and the prediciton
	'''

	while(True):

		'''
			Get data (images )
		'''
		label, img = generate_samples(1)
		img_test = img.reshape(1, rows, cols, 1)
		label_test = np_utils.to_categorical(label, classes)

		'''
			Predict whats in the image with our neural net
		'''
		score = model.predict_on_batch(img_test)
		pred = vis_dictionary[score[0].argmax()]

		'''
			create fancy window pop up
		'''
		img2 = np.zeros((50, 100, 1), dtype='uint8')
		cv2.putText(img2, str(pred), (2,45), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
		cv2.putText(img2, "PREDICTION: ", (2,25), cv2.FONT_HERSHEY_PLAIN, 1, 255,1)
		img = np.array(img * 255, dtype='uint8').reshape(rows, cols, channels)
		img2 = np.concatenate((img, img2), axis=0)

		'''
			Show prediction with fancy image pop up, 'q' to quit
		'''
		for row in img[::4]:
			string = ''
			for c in row[::2]:
				c = c[0]
				if c == 0:
					c = ' '
				else:
					c = '█'
				string += c
			print(string)
		print('Prediction: ', str(pred))

		break

if __name__ == '__main__':
	if len(sys.argv) == 2 and sys.argv[1] == 't':
		train()
	elif len(sys.argv) == 2 and sys.argv[1] == 'c':
		classify()
	else:
		print("please specify if you want to 'train : t' or 'classify: c' ")
