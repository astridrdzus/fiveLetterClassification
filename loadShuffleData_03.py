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
from getData_02 import*
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import  matplotlib.pyplot as plt
from random import shuffle
import random
import csv

def load_TrainData(num_img):
    #Getting input data
    #train_X: 1500 np.arrays of 1024 by 1 pixels
    #train_X_32: 1500 np.arrays of 32 by 32 pixels
    train_X, train_X_32= getTrainData()                             #input train images
    #print(train_X.shape)
    train_y= getLabelsTRAIN()                                       #corresponding train labels
    #print(train_y.shape)


    #-----------------------Changing every str element to a float element in every array
    train_X_float = train_X.astype(np.float)


    #----------- scaling values to a range of 0 to 1------------------------------#

    train_X_float = train_X_float/255.0

    train_X_32 = np.array(train_X_32)                      #converting to a numpy array
    #print(train_X_32.shape)

    train_X_32 = train_X_32/255.0
    #print(train_X_32)

    #----------Getting classes-----------------#
    class_names = getClasses_train()
    #print(class_names)


    #---------------------Mapped dataset----------------------------------------------------------------------#
    mapped_data = []
    element=[]
    for i in range(num_img):
        element.append(train_X_32[i])                                     #image
        element.append(train_y[i])                                        #label
        element.append(class_names[i])                                    #class

        mapped_data.append(element)
        element=[]

    return  mapped_data

def load_valData(num_img):
    #Getting input data
    #val_X: 1500 np.arrays of 1024 by 1 pixels
    #val_X_32: 1500 np.arrays of 32 by 32 pixels

    val_X, val_X_32 = getValidationData()  # Cross Validation set
    # print(val_X.shape)
    val_y = getLabelsCV()
    # print(val_y.shape)

    #-----------------------Changing every str element to a float element in every array--------#
    val_X_float = val_X.astype(np.float)

    #----------- scaling values to a range of 0 to 1------------------------------#
    val_X_float = val_X_float / 255.0
    # print(val_X_float)

    val_X_32 = np.array(val_X_32)                           # converting to a numpy array
    # print(train_X_32.shape)

    val_X_32 = val_X_32 / 255.0
    # print(train_X_32)

    # ----------Getting classes-----------------#
    class_names = getClasses_validation()
    # print(class_names)

    # ---------------------Mapped dataset----------------------------------------------------------------------#
    mapped_data = []
    element = []
    for i in range(num_img):
        element.append(val_X_32[i])  # image
        element.append(val_y[i])  # label
        element.append(class_names[i])  # class

        mapped_data.append(element)
        element = []

    return mapped_data

def load_testData(num_img):
    #Getting input data
    #val_X: 1500 np.arrays of 1024 by 1 pixels
    #val_X_32: 1500 np.arrays of 32 by 32 pixels

    test_X, test_X_32 = getTestData()  # Test Validation set
    # print(val_X.shape)
    test_y = getLabelsTRAIN()
    # print(val_y.shape)

    #-----------------------Changing every str element to a float element in every array--------#
    test_X_float = test_X.astype(np.float)

    #----------- scaling values to a range of 0 to 1------------------------------#
    test_X_float = test_X_float / 255.0
    # print(test_X_float)

    test_X_32 = np.array(test_X_32)                           # converting to a numpy array
    # print(train_X_32.shape)

    test_X_32 = test_X_32 / 255.0
    # print(train_X_32)

    # ----------Getting classes-----------------#
    class_names = getClasses_test()
    # print(class_names)

    # ---------------------Mapped dataset----------------------------------------------------------------------#
    mapped_data = []
    element = []
    for i in range(num_img):
        element.append(test_X_32[i])  # image
        element.append(test_y[i])  # label
        element.append(class_names[i])  # class

        mapped_data.append(element)
        element = []

    return mapped_data


def w_shuffle_data(mapped_data,npfile,csvfile ):

    #print(mapped_data[0])

    #*******************************************************************************************************#
    #----------------------------Suffle dataset-------------------------------------------------------------#

    new_shuffle = mapped_data
    shuffle(new_shuffle)

    #------------------------------Saving 32x32 arrays to a file--------------------------------------------#

    #save_list_images= new_shuffle[0][0]
    save_imArr_32= [row[0] for row in new_shuffle]                       #getting only the 32x32 narrays
    #print(save_imArr_32)

    save_imArr_32= np.array(save_imArr_32)
    #print(save_imArr_32[0][14])

    #------------------------------------Write csvfiles-----------------------------------------------------------#
    np.save(npfile,save_imArr_32 )


    with open(csvfile, "w+") as output:
        writer = csv.writer(output)
        writer.writerows(new_shuffle)



def r_shuffle_data(npfile, csvfile,num_img):

    #************************************************************************************************************#
    #---------------------------------------Read csvfiles--------------------------------------------------------#

    #---------------Reading labels and classes-------------

    with open(csvfile, 'r') as f:
      reader = csv.reader(f)
      new_shuffle = list(reader)

    #---------------Reading nparray images------
    images = np.load(npfile)

    #-----------Saving all together---------
    labels =  [row[1] for row in new_shuffle]
    labels = [int(i) for i in labels]
    classes = [row[2] for row in new_shuffle]

    #print(images[0][14])

    mapped_data_file= []

    element=[]
    for i in range(num_img):
        element.append(images[i])                                     #image
        element.append(labels[i])                                        #label
        element.append(classes[i])                                    #class

        mapped_data_file.append(element)
        element=[]

    return mapped_data_file

def plot_dataset(mapped_data_file, num_img):
    ######################################################################################################################3
    # ----------------------Display the first 25 images from the training set and
    #                      display the class name below each image.-----------------------------------------#
    #---------------------Plotting--------------------------------------------------------------------------#
    fig = plt.figure(1,figsize=(10,10))
    for i in range(25):                                                 #first 25 images
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.imshow(mapped_data_file[i][0], cmap=plt.cm.binary)
        plt.xlabel(mapped_data_file[i][2])
    #plt.show()

    #--------Plotting a random image and its different values------------------------------------------------#
    shuffle_number = random.randint(0,num_img)
    fig=plt.figure(2)
    plt.imshow(mapped_data_file[shuffle_number][0])
    plt.colorbar()
    plt.grid(True)
    plt.show()



    #input('presion cualquier tecla para terminar...')
    plt.close('all')

def plot_image(i, predictions_array, true_label, img):
    class_names= ['a','e','i','o','u']
    #value= 1
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    #print('true label :', true_label)
    predicted_label = np.argmax(predictions_array)
    #print('predicted label: ', predicted_label)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        #value= 0


    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)




def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')



