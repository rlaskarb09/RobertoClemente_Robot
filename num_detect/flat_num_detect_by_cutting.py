import cv2
import numpy as np

import line_detect.geom_util as geom
from bright.bright_function import edge_enhancement
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
from num_detect.detect_function import detect, warp

# datafolder = '/Users/soua/Desktop/Project/sterling_demo'
# imgpath = datafolder + '/Img_1077.jpg' # 201

datafolder = '/Users/soua/Desktop/Project/sterling_demo2'
imgpath = datafolder + '/Img_51.jpg'
# imgpath = datafolder + '/Img_45.jpg' # 102
# imgpath = datafolder + '/Img_131.jpg' # 203

img = cv2.imread(imgpath)
img = edge_enhancement(img)
cv2.imshow('origin', img)

alpha = 1.5
beta = 30
res = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
cv2.imshow('scaling img', res)
# cv2.waitKey(0)

hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv', hsv)
h, s, v = cv2.split(hsv)
v = 255 - v
s = s + 50 # 102
hsv = cv2.merge((h, s, v))
cv2.imshow('alpha beta hsv', hsv)
# cv2.waitKey(0)
# boundaries = [([60, 90, 80], [90, 255, 110])]  #GREEN
# boundaries = [([50, 80, 0], [90, 150, 20])] # sterling_demo 201
boundaries = [([40,80,0], [70,255,80])] # sterling_demo2 102

lowerG = np.array(boundaries[0][0], dtype = 'uint8')
upperG = np.array(boundaries[0][1], dtype = 'uint8')

kernel = np.ones((3,3), np.uint8)
mask = cv2.inRange(hsv, lowerG, upperG)
cv2.imshow('mask', mask)
# cv2.waitKey(0)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.waitKey(0)

img_copy = img.copy()
cnts, max_cont, approx_cnt = geom.find_main_contour_approx(mask)
cv2.drawContours(img_copy, max_cont, -1, (255, 0, 0), 2)
cv2.imshow('first contour', img_copy)
cv2.drawContours(img_copy, [approx_cnt], -1, (0, 0, 255), 2)
cv2.imshow('approx contour', img_copy)

peri = cv2.arcLength(approx_cnt, True)
approx = cv2.approxPolyDP(approx_cnt, 0.02*peri, True)
if len(approx) == 4:
    our_cnt = approx
#
warp = warp(our_cnt, img)
cv2.imshow('warp image', warp)

warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
cv2.imshow('warp gray', warp_gray)
# warp_gray= edge_enhancement(warp_gray)
thresh_ = cv2.threshold(warp_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow('thresh', thresh_)
th_kernel = np.ones((2,2), np.uint8)
thresh = cv2.morphologyEx(thresh_, cv2.MORPH_OPEN, th_kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, th_kernel)
# cv2.imshow('thresh after morpho', thresh)

thresh_cut = []
for i in range(3):
    cut_size = thresh.shape[1]//3
    if i == 2:
        cut_img = thresh[:, i*cut_size:]
    else:
        cut_img = thresh[:, i*cut_size : (i+1)*cut_size]
    thresh_cut.append(cut_img)
cv2.imshow('first digit', thresh_cut[0])
cv2.imshow('second digit', thresh_cut[1])
cv2.imshow('third digit', thresh_cut[2])
cv2.waitKey(0)

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (1, 1, 1, 1, 1, 1, 1): 0,
    # (1, 1, 1, 1, 1, 1, 1): 1,
    (0, 0, 1, 0, 0, 1, 0): 1,

    (1, 0, 1, 1, 1, 0, 1): 2,
    (0, 0, 1, 1, 1, 0, 1): 2,

    (1, 0, 1, 1, 0, 1, 1): 3
}

digits = []
for i in range(3):
    cut_img = thresh_cut[i]
    cv2.imshow('%d th image'%i, cut_img)
    each_cnts, max_cont, box = geom.find_main_contour_box(cut_img.copy())
    cv2.imshow('find contour of each digit', cut_img)
    cv2.waitKey(0)
    (x, y, w, h_) = cv2.boundingRect(max_cont)
    if i == 0:
        if w < 10:
            digits.append(1)
        else:
            digits.append(2)
    elif i == 1:
        digits.append(0)
    elif i == 2:
        if w < 10:
            digits.append(1)
        else:
            roi = cut_img[y:y+h_, x:x+w]
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)
            segments = [
                ((0, 0), (w, dH)),  # top
                ((0, dH), (dW, h_ // 2)),  # top-left
                ((w - dW, dH), (w, h_ // 2)),  # top-right
                ((0, (h_ // 2) - dHC), (w, (h_ // 2) + dHC)),  # center
                ((0, h_ // 2), (dW, h_)),  # bottom-left
                ((w - dW, h_ // 2), (w, h_)),  # bottom-right
                ((0, h_ - dH), (w, h_))  # bottom
            ]
            on = [0] * len(segments)
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                if total / float(area) > 0.60:
                    on[i] = 1
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append(digit)

print("digits: {}{}{}".format(*digits[:3]))
cv2.imshow("Output", cv2.resize(thresh, (250, 250)))
cv2.waitKey(0)

#
