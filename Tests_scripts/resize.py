import cv2

img = cv2.imread('/home/asteroid/PycharmProjects/nnVocales/AEIOU_prepared/A/0.Training_A/a_train_1.png', cv2.IMREAD_GRAYSCALE)

print('Original Dimensions : ',img.shape)

scale_percent = 25 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
