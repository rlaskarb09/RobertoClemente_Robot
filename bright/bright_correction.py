import numpy as np
import cv2
import num_detect.number_geom as geom

from bright.bright_function import edge_enhancement

#################
datafolder = '/Users/soua/Desktop/Project/sterling_demo'
img_path = datafolder + '/Img_89.jpg' # 202 s : 56 -> +50 / v : 104 -> 255 - = 151
# img_path = datafolder + '/Img_284.jpg' # 203 s : 84 -> +50 / v : 81 -> 255 - = 175
# img_path = datafolder + '/Img_390.jpg' # 103 s : 41 -> + 100 / v : 141 -> 255 - = 115
# img_path = datafolder + '/Img_907.jpg' # 101 s : 90 / v : 71 -> 235 - = 164
# img_path = datafolder + '/Img_1398.jpg' # 101 s : 78 / v : 69
# img_path = datafolder + '/Img_1077.jpg' # 201 s : 35 / V : 138

img = cv2.imread(img_path)
img = edge_enhancement(img)
cv2.imshow('before scaling', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_size = len(hist)

accm = []
accm.append(float(hist[0]))
for idx in range(1, hist_size):
    accm.append(accm[idx-1] + float(hist[idx]))

max = accm[-1]
clip_hist_percent = 1
clip_hist_percent *= (max/100.0)
clip_hist_percent /= 2.0

min_gray = 0
while accm[min_gray] < clip_hist_percent:
    min_gray += 1
max_gray = hist_size - 1
while accm[max_gray] >= (max - clip_hist_percent):
    max_gray -= 1

# alpha = 255 / (max_gray - min_gray)
# beta = - min_gray * alpha
alpha = 1.95
beta = 0

res = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)
cv2.imshow('after scaling', res)
hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
print('v average : ', np.mean(v))
print('s average : ', np.mean(s))
# v = 235 - v
# s = s
# hsv = cv2.merge((h, s, v))

cv2.imshow('after hsv', hsv)
# img_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# cv2.imshow('hsv img', img_hsv)
# cv2.waitKey(0)

boundaries = [([65, 180, 80], [80, 255, 110])]  #GREEN

lowerG = np.array(boundaries[0][0], dtype = 'uint8')
upperG = np.array(boundaries[0][1], dtype = 'uint8')
# mask = cv2.inRange(hsv, lowerG, upperG)
# kernel = np.ones((3,3),np.uint8)
# morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
mask = cv2.inRange(hsv, lowerG, upperG)
# output = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('mask', mask)
# cv2.imshow('output', output)
# cv2.imshow('morph', morph)
# cv2.waitKey(0)
# output = cv2.bitwise_and(img, img, mask = morph)
cnts, max_cont, box = geom.find_main_contour(mask)
c = max(cnts, key = cv2.contourArea)
# cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# c = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
cv2.drawContours(img, c, -1, (255, 0, 0), 2)
cv2.imshow('contour', img)
cv2.waitKey(0)

mark = cv2.minAreaRect(c)
box = cv2.boxPoints(mark)
box = np.int0(box)
cv2.drawContours(img, [box], -1, (0, 0, 255), 2)
cv2.imshow('box', img)
cv2.waitKey(0)