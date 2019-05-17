"""
__________________________
Input             | Output
__________________________
1 1 1 0 0 1 1 1 1 |  1
1 0 0 0 1 0 0 0 0 |  0
1 1 0 0 1 1 0 1 0 |  0
1 0 0 1 0 1 0 0 0 |  0
1 0 0 1 1 0 1 0 1 |  1
0 0 1 0 0 0 0 1 1 |  1
__________________________  . . . and so on.

"""

#We set the seed value with a constant number to get consistent #results every time our neural network is trained.
#This is because when we create a model in Keras, it is assigned #random weights every time. Due to this, we might receive different #results every time we train our model.
#However this won't be a general issue, our training set is small, #so we take this precaution
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import keras

#We create a training set of fifteen 8-bit binary numbers
train_X = np.array([[0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1],
                    [0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,1,1],
                    [0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0,1],
                    [0,0,0,0,0,1,1,0],
                    [0,0,0,0,0,1,1,1],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,1],
                    [0,0,0,0,1,0,1,0],
                    [0,0,0,0,1,0,1,1],
                    [0,0,0,0,1,1,0,0],
                    [0,0,0,0,1,1,0,1],
                    [0,0,0,0,1,1,1,0],
                    [0,0,0,0,1,1,1,1]])
train_y = np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
test_X = np.array([[1,0,0,1,1,1,0,1], [1,0,0,0,0,1,1,0]])

#We create a Sequential model
model = keras.models.Sequential()

#In the next line, we add both the required layers to our model
#The input_dim parameter is the number of input neurons and the #first parameter,
# "1" is the number of neurons in our first hidden #layer(which happens to be the output layer)
model.add(keras.layers.Dense(1, input_dim=8, activation='sigmoid'))

#Configuring the model for training
model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
              loss='mse', metrics=['accuracy'])


#We have defined all parameters, so we train our model with the training data we created.
#epochs: Number of times the dataset is passed through the network
#batch_size: Number of samples passed at a time
model.fit(train_X, train_y, epochs= 20, batch_size= 5)


#Our Neural Network has been trained now. We pass two inputs to the #model to check the results it produce.
#The results would be in the range of 0.98 or 0.02 cause we might #still be left with some error rate(which could've been fixed if we #used a bigger training set or different model parameters.
#Since we desire binary results, we just round the results using the #round function of Python.
#The predictions are actually in
print([round(x[0]) for x in model.predict(test_X)])