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
from keras.models import load_model



def runModel():                                       #Building a simple model

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32)),
        keras.layers.Dense(128, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])


    '''
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32)),
        keras.layers.Dense(1024, activation=tf.nn.sigmoid),
        keras.layers.Dense(1024, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])
    '''

    return model

#------------Train the model---------------------#
def training(modelFile):
    model.compile(optimizer= 'Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #model.fit(train_images, train_labels, batch_size=50, epochs=500)
    #model.fit(train_images, train_labels, batch_size=10, epochs=1000)
    #model.fit(train_images, train_labels, batch_size=1, epochs=1000)
    model.fit(train_images, train_labels, batch_size=50, epochs=1000)


    # ----------------------------Saving the model-------------------------------------#
    model.save(modelFile)

def loadModel(modelFile):
    # -------------------Load model------------------------#
    model = keras.models.load_model(modelFile)
    return  model

def predict(model, images,labels ):

    # ------------------Evaluate accuracy----------------------#
    test_loss, test_acc = model.evaluate(images, labels)

    print('Test accuracy:', test_acc)
    predictions = model.predict(images)

    return predictions

def dispPredict(model, predictions, num_img):       #Display Prediction
    map_pred= []

    #predictions= predictions*10
    label_0 = val_labels[num_img]
    classes_0 = val_classes[num_img]

    #print(len(predictions[0]))

    predictions[num_img]= np.array(predictions[num_img])
    print(predictions[num_img])
    #print(np.argmin(predictions[0]))

    print('label: ', label_0)
    print('class: ', classes_0)
    print('prediction: ', np.argmax(predictions[num_img]))

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    predict = plot_image(num_img, predictions, val_labels, val_images)
    plt.subplot(1, 2, 2)
    plot_value_array(num_img, predictions, val_labels)
    plt.show()

    map_pred.append(num_img)
    map_pred.append(predict)
    return map_pred


def saveWrongPred():
    n = random.randint(0,499)
    print(n)
    result=[]
    flag= 1
    contFalse= 0                                    #Counts the number of wrong predictions
    wrongPImg= []                                   #Saves the index of the wrong predictions
    for i in range(0,500):
        #print('n: ', i)
        if np.argmax(predictions[i]) != val_labels[i]:
            flag=0
            contFalse= contFalse+1
            wrongPImg.append(i)
        result.append(flag)
        flag=1
        
    # print('Number of wrong predictions: ', contFalse)
    # print('Images of wrong predictions: ', wrongPImg)
    return wrongPImg


#----------------------Train dataset--------------------------------------------#

fileCSV_train= 'shuffle_train.cvs'
fileNPY_train= 'shuffleIMG_train.npy'
numImg_train = 1500

#------------ shuffle-------------------------------------------------------------#
#trainData= load_TrainData(numImg_train)
#wShuffle_train= w_shuffle_data(trainData, fileNPY_train,fileCSV_train)

dataset_train= r_shuffle_data(fileNPY_train,fileCSV_train,numImg_train)

#------------Plotting dataset train----------------------------------------------#
#plot_dataset(dataset_train, numImg_train)

train_images= [row[0] for row in dataset_train]
train_labels= [row[1] for row in dataset_train]
train_classes= [row[2] for row in dataset_train]

train_images= np.array(train_images)
print(train_images.shape)
#print(train_classes[0])

#---------------------Validation dataset------------------------------------------#

fileCSV_val= 'shuffle_val.cvs'
fileNPY_val= 'shuffleIMG_val.npy'
numImg_val= 500
#------------ shuffle-------------------------------------------------------------#
#valData= load_valData(numImg_val)
#wShuffle_val = w_shuffle_data(valData, fileNPY_val, fileCSV_val)

dataset_val= r_shuffle_data(fileNPY_val,fileCSV_val,numImg_val)

#------------Plotting dataset val----------------------------------------------#
#plot_dataset(dataset_val, numImg_val)

val_images= [row[0] for row in dataset_val]
val_labels= [row[1] for row in dataset_val]
val_classes= [row[2] for row in dataset_val]

val_images= np.array(val_images)
print(val_images.shape)
#------------------------------------------------------------------------------------#


#modelFile= 'model_arch.h5'
#modelFile= 'model_arch_2.h5'
#modelFile= 'model_arch_3.h5'
modelFile= 'model_arch_4.h5'

model= runModel()
training(modelFile)


#model = loadModel(modelFile)
predictions= predict(model, val_images,val_labels)
wrongPImg= saveWrongPred()

print(wrongPImg)
'''
#---------------------------Displaying wrong predictions
for item in wrongPImg:
    dispPredict(model,predictions, item)

'''



