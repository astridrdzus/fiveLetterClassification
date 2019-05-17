import PIL
import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt
from numpy import asarray
from numpy import array
import cv2
'''
#load the image
img = Image.open('e0000001.png')
#summarize some details about the image
print(img.format)
print(img.mode)
print(img.size)
#show the image
img.show()
'''


'''
# load and display an image with Matplotlib
# load image as pixel array
data = image.imread('e0000001.png')
# summarize shape of the pixel array
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
pyplot.imshow(data)
pyplot.show()
'''

'''
# load image and convert to and from NumPy array
from PIL import Image
from numpy import asarray
# load the image
image = Image.open('opera_house.jpg')
# convert image to numpy array
data = asarray(image)
# summarize shape
print(data.shape)
# create Pillow image
image2 = Image.fromarray(data)
# summarize image details
print(image2.format)
print(image2.mode)
print(image2.size)
'''
'''
# load image and convert to and from NumPy array
# load the image
#image = Image.open('e0000001.png')
img = Image.open('e0000001.png').convert('LA')
img.save('greyscale.png')
image = Image.open('greyscale.png')
# convert image to numpy array
data = asarray(image)
# summarize shape
print(data.shape)
print(type(data))
print(data.size)
print(data[0])
'''
'''
# create Pillow image
image2 = Image.fromarray(data)
# summarize image details
print(image2.format)
print(image2.mode)
print(image2.size)
'''

'''
#converts an image from rgb values to grayscale values
img = cv2.imread('e0000001.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# convert image to numpy array
data = asarray(gray)

#save the pixel values to an uniarray (list)
uni= []
for i in range(128):
    for j in range(128):
        uni.append(data[i][j])


arraux = []
arraux.append(uni)
arraux.append("hello0")
#np.savetxt('dataset.txt', uni)


#np.savetxt('dataset.txt', arraux)
f= open("dataset.txt","w+")
for i in arraux:
    f.write(str(i)+"\n")
f.close()
'''


img = cv2.imread('/home/asteroid/PycharmProjects/nnVocales/AEIOU_prepared/A/0.Training_A/a_train_1.png', cv2.IMREAD_GRAYSCALE )
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()