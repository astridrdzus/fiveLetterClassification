##.-*-coding: utf-8-*---------------------------------------------------------------------------
# Facultad de Matemáticas de la Universidad Autónoma de Yucatán
#--------------------------------------------------------------------------
#
# MODULE: Obtaning image information
# FILE: getData02.py
# USE: Preparing data
#
#> @author
#> Astrid Giselle Rodriguez Us
#
#  DESCRIPTION:
# This script reads a txt file to convert the data into numpy arrays
# REVISION HISTORY:
#
# 02 April 2019 - Initial Version
# -- April 2019 - Final Version
# TODO_dd_mmm_yyyy - TODO_describe_appropriate_changes - TODO_name
#
# Modifications:
#03 April 2019 - implemented getLabel function
#29 April 2019 - linked with extractData01_2.py and implemented
#               -getTrainData()
#               -getValidationData()
#01 May 2019    - implemented getClasses function
#--------------------------------------------------------------------------


import string
import numpy as np
from extractData_01 import *
def convert(file, num_i):                         #Read the file of images and converts it
    f = open(file,'r')                            #to numpy array
    images= []
    image_list=[]

    for i in range(num_i):                         #Indicate the number of images
        l = f.readline()
        #print(l)
        image_list = l.strip(']\n[').split(', ')  #Save the line into a list
        images.append(image_list)
        image_list = []

    im_array = np.array(images)                   #Converts the list into a numpy array
    #print(im_array.shape)                         #Verify the number of rows an columns

    #im_array = im_array.astype(np.int_)           #Converts each string item to an int item
    #image_list = map(int, image_list)             # Converts each string item to an int item
    return im_array
    #print(type(image_list))
    #return  image_list

def getLabelsTRAIN():
    aeiou_label = [0]*300               #a label
    aeiou_label = aeiou_label+[1]*300   #e label
    aeiou_label = aeiou_label+[2]*300   #i label
    aeiou_label = aeiou_label+[3]*300   #o label
    aeiou_label = aeiou_label+[4]*300   #u label
    label_array = np.array(aeiou_label)           #Converts the list into a numpy array
    #print(label_array.shape)
    return label_array

def getLabelsCV():
    # Fills the array with the vector v, n times
    aeiou_label = [0]*100               #a label
    aeiou_label = aeiou_label+[1]*100   #e label
    aeiou_label = aeiou_label+[2]*100   #i label
    aeiou_label = aeiou_label+[3]*100   #o label
    aeiou_label = aeiou_label+[4]*100   #u label
    label_array = np.array(aeiou_label)           #Converts the list into a numpy array
    #print(label_array.shape)
    return label_array
    #return  aeiou_label


def getTrainData():
    #Directory where the images are stored
    dir = '/home/asteroid/PycharmProjects/nnVocales/dataset_train/data_train_'
    filename = 'datasetAEIOU_TRAIN.txt'
    num_img = 1500                                 #number of images in the directory
    img32Arr= extractData(dir,num_img,filename)

    dataTrain_Array = convert(filename,num_img)
    return dataTrain_Array, img32Arr

def getValidationData():
    #Directory where the images are stored
    dir = '/home/asteroid/PycharmProjects/nnVocales/dataset_cross_validation/data_cv_'
    filename = 'datasetAEIOU_CV.txt'
    num_img = 500                                  #number of images in the directory
    img32Arr= extractData(dir,num_img,filename)

    dataCV_Array = convert(filename,num_img)
    return dataCV_Array, img32Arr

def getTestData():
    #Directory where the images are stored
    #dir = '/home/asteroid/PycharmProjects/nnVocales/dataset_test/data_test_'
    dir = '/home/asteroid/PycharmProjects/nnVocales/dataset_test/data_test_'
    filename = 'datasetAEIOU_TRAIN.txt'
    num_img = 500                                  #number of images in the directory
    img32Arr= extractData(dir,num_img,filename)

    dataTest_Array = convert(filename,num_img)
    return dataTest_Array, img32Arr

def getClasses_train():
    class_names= ['a']*300                          #a class
    class_names= class_names+ ['e']*300             #e class
    class_names = class_names + ['i'] * 300         #i class
    class_names = class_names + ['o'] * 300         #o class
    class_names = class_names + ['u'] * 300         #u class

    return  class_names

def getClasses_validation():
    class_names= ['a']*100                          #a class
    class_names= class_names+ ['e']*100            #e class
    class_names = class_names + ['i'] * 100         #i class
    class_names = class_names + ['o'] * 100         #o class
    class_names = class_names + ['u'] * 100         #u class

    return  class_names

def getClasses_test():
    class_names= ['a']*100                          #a class
    class_names= class_names+ ['e']*100            #e class
    class_names = class_names + ['i'] * 100         #i class
    class_names = class_names + ['o'] * 100         #o class
    class_names = class_names + ['u'] * 100         #u class

    return  class_names

