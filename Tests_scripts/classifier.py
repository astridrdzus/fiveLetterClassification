from getData02 import*
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import keras

train_X = convert('datasetAEIOU.txt',1500)         #input images
train_y = getLabels()                         #corresponding labels
test_X = convert('datasetAEIOU_CV.txt',25)       #Cross Validation set
#train_y = map(int, train_y)
print(type(train_y))
print(train_y[1])
print(train_y[1][0])
print(type(train_y[1][1]))
#We create a Sequential model
model = keras.models.Sequential()

#Configuring the model for training
model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
              loss='mse', metrics=['accuracy'])

model.add(keras.layers.Dense(1025, input_dim=1024, activation='sigmoid'))
model.add(keras.layers.Dense(5, input_dim=1025, activation='sigmoid'))
model.fit(train_X, train_y, epochs= 20, batch_size= 500)
print('finish   ')

#print([round(x[0]) for x in model.predict(test_X)])

print([x[0] for x in model.predict(test_X)])