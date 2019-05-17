from getData02 import convert, getLabels
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from math import*

from preprocessing import *
from plotting import *

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

def sigmoid_prime(x):
    return (1 / (1 + np.e ** -x))*(1-(1 / (1 + np.e ** -x)))

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 12, 6
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#Setting parameters
epochs = 50000
#input_size = 16384                             #total number of pixels
input_size = 2                                  #Example value
#hidden_size = 16385
hidden_size= 3                                  #Example value
#output_size = 2
output_size= 1                                  #Example value
LR = .1                                         #Learning Rate

#Setting the data
#x = convert('datasetEO.txt')                    #Gets the numpy array of the images
#y = getLabels()                                 #Gets the numpy array labels of the images
#print(x.shape, y.shape)
x = np.array([[0,0], [0,1], [1,0], [1,1]])       #Example input data
y = np.array([ [0],   [1],   [1],   [0]])       #Example input labels

#initialize the weights to the neural network to random numbers
w_hidden = np.random.uniform(size= (input_size,hidden_size))
#print(w_hidden.shape)
#print(w_hidden)
w_output = np.random.uniform(size = (hidden_size, output_size))
#print(w_output.shape)
#pr#int(w_output)

#Backpropagation algorithm

for epoch in range(epochs):
    #Forward
    act_hidden = sigmoid(np.dot(x,w_hidden))     #dot matrix multiplication
    #print("act_hidden: ", act_hidden)
    output = np.dot(act_hidden, w_output)

    #Calculate error
    error = y - output

    if epoch % 5000 == 0:
        print(f'error sum {sum(error)}')

    #Backward
    dZ = error * LR
    w_output += act_hidden.T.dot(dZ)
    dH = dZ.dot(w_output.T)* sigmoid_prime(act_hidden)
    w_hidden += x.T.dot(dH)


x_test = x[1] # [0, 1]

act_hidden = sigmoid(np.dot(x_test, w_hidden))
print(np.dot(act_hidden, w_output))
