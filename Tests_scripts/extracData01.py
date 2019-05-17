##.-*-coding: utf-8-*---------------------------------------------------------------------------
# Facultad de Matemáticas de la Universidad Autónoma de Yucatán
#--------------------------------------------------------------------------
#
# MODULE: Image information extraction
# FILE: extractData01.py
# USE: Preparing data
#
#> @author
#> Astrid Giselle Rodriguez Us
#
#  DESCRIPTION:
# This script reads images of 128x128 pixels, obtains their value data and saves them
# to a txt file.
#
# REVISION HISTORY:
#
# 02 April 2019 - Initial Version
# -- April 2019 - Final Version
# TODO_dd_mmm_yyyy - TODO_describe_appropriate_changes - TODO_name
#
# Modifications:
#03 April 2019 - number of lines at the end of the process
#22 April 2019 - resize images to an 25% of the original image
#--------------------------------------------------------------------------

from numpy import asarray
import cv2

image_list = []
for file in range(1500):                             #Indicate the number of images
    img= cv2.imread('/home/asteroid/PycharmProjects/nnVocales/dataset_train/data_train_'+str(file+1)+'.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #converts an image from rgb values to grayscale values
    #print('Original Dimensions : ', gray.shape)
    scale_percent = 25                              # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

    #print('Resized Dimensions : ', resized.shape)


    data = asarray(resized)                            # convert image to numpy array
    #print(data)

    unidim = []                                     #save the pixel values to an uniarray list
    for i in range(32):
        for j in range(32):
            unidim.append(data[i][j])
    image_list.append(unidim)

#save each image list to a file
f= open("datasetAEIOU.txt","w+")
for i in image_list:
    f.write(str(i)+"\n")                            #each image list in a line
f.close()
file = open("datasetAEIOU.txt","r")
print(len(file.readlines()))                        #conts the total number of lines
file.close()



