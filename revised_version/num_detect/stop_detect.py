import cv2
import numpy as np

from bright.bright_function import edge_enhancement
import line_detect.geom_util as geom
from num_detect.detect_function import detect

datafolder = '/Users/soua/Desktop/Project/sterling_demo2'
imgpath = datafolder + '/Img_580.jpg'

img = cv2.imread(imgpath)
img = edge_enhancement(img)
cv2.imshow('origin', img)

alpha = 1.5
beta = 50
res = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
cv2.imshow('convert scaling', res)

hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv', hsv)

h, s, v = cv2.split(hsv)
print('v max : %f, min : %f, mean : %f'%(np.amax(v[:]), np.amin(v[:]), np.mean(v[:])))
print('s max : %f, min : %f, mean : %f'%(np.amax(s[:]), np.amin(s[:]), np.mean(s[:])))
s = s + 50
hsv = cv2.merge((h, s, v))
cv2.imshow('alpha beta hsv', hsv)
# cv2.waitKey(0)

boundaries = [([0, 0, 255], [20, 100, 255])] # RED
lowerR = np.array(boundaries[0][0], dtype='uint8')
upperR = np.array(boundaries[0][1], dtype='uint8')

mask = cv2.inRange(hsv, lowerR, upperR)
cv2.imshow('mask', mask)
# cv2.waitKey(0)
cnts, max_cont, approx_cnt = geom.find_main_contour_approx(mask)
cv2.drawContours(img, max_cont, -1, (255, 0, 0), 2)
cv2.drawContours(img, approx_cnt, -1, (0, 255, 0), 2)
cv2.imshow('contours', img)
cv2.waitKey(0)

# shape, thresh = detect(max_cont, mask)
# print(1)
# if shape == 'octagon':
#     cv2.putText(img, "STOP", )
