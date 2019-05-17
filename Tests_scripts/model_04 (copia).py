##.-*-coding: utf-8-*---------------------------------------------------------------------------
# Facultad de Matemáticas de la Universidad Autónoma de Yucatán
#--------------------------------------------------------------------------
#
# MODULE: Building a model
# FILE: model_3.py
# USE: Build de Neural Network Arquitecture
#
#> @author
#> Astrid Giselle Rodriguez Us
#
#  DESCRIPTION:
# This script implements a neural network arquitecture using Tensorflow and Keras
# 29 April 2019 - Initial Version
# -- April 2019 - Final Version
# TODO_dd_mmm_yyyy - TODO_describe_appropriate_changes - TODO_name
#
# Modifications:
#22 April 2019 - implemented getLabel function
#01 May   2019 - added graphics to visualize the data
#--------------------------------------------------------------------------
from loadShuffleData_03 import*
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import  matplotlib.pyplot as plt
from random import shuffle
import random
import csv

dataTrain= 'shuffle_dataset.cvs'
dataImg= 'shuffle_images.npy'
dataset_train = r_shuffle_data(dataTrain, dataImg,1500)
#plot_dataset(dataset_train)

train_images= [row[0] for row in dataset_train]
train_labels= [row[1] for row in dataset_train]
train_classes= [row[2] for row in dataset_train]

train_images= np.array(train_images)
print(train_images.shape)
#print(train_classes[0])
#------------------------------------------------------------------------------------#


#Building a simple model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)



'''
#Building a simple model

model= tf.keras.Sequential()
#Adds a densely-connected layer with 1500 units to the model:
model.add(layers.Dense(1024, activation='sigmoid'))
# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.01))
#Add a softmax layer with 5 output units:
model.add(layers.Dense(5, activation="softmax"))

model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
              loss='mse',
              metrics=['mae'])
#------------------------------------------------------------------------------------#

#Evaluate and predict

model.fit(train_X, train_y, epochs=20, batch_size=50)

#model.evaluate(val_X, labels, batch_size=32)

#model.evaluate(dataset, steps=30)
print(val_X[0].shape)
model.evaluate(val_X[0], steps=50)

'''