#AMANZHOL DARIBAY ID201416749 NAZARBAYEV UNIVERSITY
#FULL CODE with Explanation

#STEP1: Importing libraries and MNIST data

import cv2 as cv #for displaying it's been used OpenCV
import numpy as np #for array-handling and plotting
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt #for plotting


import os #for saving the data in folder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #keep keras backend tensorflow quiet


#importing from keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dropout, Activation
from keras.layers import Dense
from keras.utils import np_utils
import h5py #for creation h5py data

#now we need to split MNIST data into training and testing data
#MNIST contains 70,000 images of handwritten digits: 60,000 for 
#training and 10,000 for testing. The images are grayscale, 
#28x28 pixels, and centered to reduce preprocessing and get started quicker.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#MNIST dataset contains only grayscale images 
#we can view them as follows using OpenCV
#it needs to be uncommented
#numpy_horizontal = np.hstack((X_train[1], X_train[2], X_train[3], X_train[4], X_train[5]))
#cv.imshow("Image", numpy_horizontal)
#cv.waitKey(0)



#STEP2: Preprocessing the data
#In order to see the shapes of downloaded data
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_train shape", X_test.shape)
print("y_test shape", y_test.shape)

#In order to input, it's been reshaped the actual data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
#For convenience it's been converted the type of data from uint8 to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#Again for the input, it's been normalized
X_train /= 255
X_test /= 255

#Shape after two steps of reshape and normalization
print("After preprocessing X_train shape is", X_train.shape)
print("After preprocessing X_test shape is", X_test.shape)

#To show the classes (labels) that are from 0 to 9
#it's been used np.unique function, which returns 
#the stored uniques values (not all of the values within y_train)
#and it's type as uint8 also if return_counts is True, specifies 
#the occurance of respective number in the array
print(np.unique(y_train, return_counts = True))


#Then it's been done one-shot-encoding becuase
#the classes should have the same meaning for computer
#not 0<1<2<3, thus y values will be encoded into the
#vectors with the same meaning
#the encoding has been done using keras' numpy-related utilities
num_classes = 10
print("Shape before OHE:", y_train.shape)
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)
print("Shape after OHE:", Y_train.shape)



#STEP3: Building the Network
#keras is a library which helps to build a neural network
#print("NETWORK ARCHITECTURE is Keras Sequential Model,")
#print("where layers can be stacked by .add() method")
#print("The INPUT LAYER needs to be specified,")
#print("Input uints are 784")
#print("Number of hidden layers are 2")
#print("Number of nuerons in hidden layers are 512")
#print("It should not be simple binary classification at neurons,")
#print("thus it has been used ACTIVATION FUNCTIONs in hidden layers")
#print("The differentiation for the training via backpropagation is happening")
#print("behind without having to specified details")
#print("In order to prevent overfitting, it's been used dropout,")
#print("which keeps some weights randomly assigned")
#print("The OUTPUT UNITS are specified as 10 and the FUNCTION used")
#print("is a softmax, which is standard for multi-class classification")

model = Sequential()
model.add(Dense(512, input_shape=(784,))) #The input layer and the first hidden layer
model.add(Activation('relu')) #the fucntion used in first hidden layer
model.add(Dropout(0.2)) #to avoid overfitting

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))



#STEP4: Compilation and Training the Model
#print("Learning process ...")
#print("Loss function is Categorical Cross Entropy")
#print("Optimizer is Adam")
#print("Metric is Accuracy")
#print("Iterations (epochs) are 20")
#print("Samples per update (batch size) is 128")

#training the model and saving validation data in history
model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
history = model.fit(X_train, Y_train,
	batch_size = 128, epochs = 70,
	verbose = 2,
	validation_data = (X_test,Y_test))

#saving the model
#model_name = h5py.File('mnist.hdf5', 'w'
save_dir = "/home/amanzhol/MNIST"
model_name = 'mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('The model is saved at %s ' % model_path)

#STEP5:The Evaluation of the Model
mnist_model = load_model('mnist.h5')
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose = 2)

#Note about verbose
#By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
#verbose=0 will show you nothing (silent)
#verbose=1 will show you an animated progress bar like this:
#progres_bar
#verbose=2 will just mention the number of epoch like this:
#Epoch 1/70

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])


