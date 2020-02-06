import cv2
import numpy as np
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
import time

datafolder = '/Users/soua/Desktop/Project/distance'
img30_path = datafolder + '/Img_4.jpg'
img30 = cv2.imread(img30_path)

# starttime = time.time()
yen_th30 = threshold_yen(img30)

# print('converting time = ', time.time()-starttime)

img15_path = datafolder + '/Img_152.jpg'
img15 = cv2.imread(img15_path)
yen_th15 = threshold_yen(img15)

bright30 = rescale_intensity(img30, (0, yen_th15), (0, 255))
bright15 = rescale_intensity(img15, (0, yen_th15), (0, 255))

hsv15 = cv2.cvtColor(bright15, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv15', hsv15)

hsv30 = cv2.cvtColor(bright30, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv30', hsv30)
cv2.waitKey(0)

imsave('out30.jpg', bright30)
imsave('out15.jpg', bright15)
