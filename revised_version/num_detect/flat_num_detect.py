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
imgpath = datafolder + '/Img_45.jpg' # 102
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
# s = s + 30
hsv = cv2.merge((h, s, v))
cv2.imshow('alpha beta hsv', hsv)
# cv2.waitKey(0)
# boundaries = [([60, 90, 80], [90, 255, 110])]  #GREEN
# boundaries = [([50, 80, 0], [90, 150, 20])] # sterling_demo 201
boundaries = [([40,80,0], [70,255,80])] # sterling_demo 102

lowerG = np.array(boundaries[0][0], dtype = 'uint8')
upperG = np.array(boundaries[0][1], dtype = 'uint8')

kernel = np.ones((3,3), np.uint8)
mask = cv2.inRange(hsv, lowerG, upperG)
cv2.imshow('mask', mask)
# cv2.waitKey(0)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# cv2.waitKey(0)

img_copy = img.copy()
cnts, max_cont, approx_cnt = geom.find_main_contour(mask)
cv2.drawContours(img_copy, max_cont, -1, (255, 0, 0), 2)
cv2.imshow('first contour', img_copy)
cv2.drawContours(img_copy, [approx_cnt], -1, (0, 0, 255), 2)
cv2.imshow('approx contour', img_copy)
# cv2.waitKey(0)
# img_copy = img.copy()
# cv2.drawContours(img_copy, [box], -1, (255, 0, 0), 2)
# cv2.imshow('box', img_copy)
#
# warp = four_point_transform(img, box)
# cv2.imshow('warp image', warp)
# cv2.waitKey(0)

# our_cnt = None
peri = cv2.arcLength(approx_cnt, True)
approx = cv2.approxPolyDP(approx_cnt, 0.02*peri, True)
if len(approx) == 4:
    our_cnt = approx
#
warp = warp(our_cnt, img)
cv2.imshow('warp image', warp)
# cv2.waitKey(0)

# shape, th_digits = detect(box, v)
# digits = cv2.findContours(th_digits.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# digits = imutils.grab_contours(digits)
# cv2.drawContours(warp, digits, -1, (255, 0, 0), 2)
# cv2.imshow('digit', warp)
# cv2.waitKey(0)
#
# digitcnts = []
# for cc in digits:
#     (x, y, w, h) = cv2.boundingRect(cc)
#     if (w >= 4 and w <= 40) and (h >= 25 and h <= 30):
#         digitcnts.append(cc)
#
#     digitcnts = contours.sort_contours(digitcnts, method='left-to-right')[0]
#     digits=[]
#     print('%d digits detected'%len(digitcnts))
warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
warp_filt = cv2.adaptiveThreshold(warp_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
cv2.imshow('warp gray', warp_gray)
cv2.imshow('warp filtered', warp_filt)
# warp_gray= edge_enhancement(warp_gray)
thresh_ = cv2.threshold(warp_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('thresh', thresh_)
# cv2.waitKey(0)
# thresh = 255 - thresh_
th_kernel = np.ones((2,2), np.uint8)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, th_kernel)
thresh = cv2.morphologyEx(thresh_, cv2.MORPH_OPEN, th_kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, th_kernel)

thresh_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
thresh_cnts = imutils.grab_contours(thresh_cnts)

cv2.imshow('thresh after morpho', thresh)
cv2.waitKey(0)

digitcnts = []
for c in thresh_cnts:
    (x, y, w, h_) = cv2.boundingRect(c)
    if (h_ >= 20 and h_ <= 40):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        digitcnts.append(box)
    # if w >= 10 and (h_ >= 20 and h_ <= 40):
    #     rect = cv2.minAreaRect(c)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     digitcnts.append(box)
    # elif w < 5 and (h_ >= 20 and h_ <= 40):
    #     rect = cv2.minAreaRect(c)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     digitcnts.append(box)

cv2.drawContours(warp, digitcnts, -1, (255, 0, 0), 2)
cv2.imshow('draw digit', warp)
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
for cc in digitcnts:
    (x, y, w, h) = cv2.boundingRect(cc)
    if w < 5:
        w = w+8
        x = x-8
    roi = thresh[y:y+h, x:x+w]
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)
    cv2.drawContours(warp, [cc], -1, (0, 0, 255), 2)
    cv2.imshow('digit', warp)
    cv2.waitKey(0)
    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, dH), (dW, h // 2)),  # top-left
        ((w - dW, dH), (w, h // 2)),  # top-right
        ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((w - dW, h // 2), (w, h)),  # bottom-right
        ((0, h - dH), (w, h))  # bottom
    ]
    # segments = [
    #     ((w // 2, 0), (w//2, dH)), # top
    #     ((0, h//4), (dW, h//4)), # top-left
    #     ((w-dW, h//4), (w, h//4)), # top-right
    #     ((w//2, h//2 - dH//2), (w//2, h//2 + dH//2)), # center
    #     ((0, 3*h//4), (dW, 3*h//4)), # bottom-left
    #     ((w-dW, 3*h//4), (w, 3*h//4)), # bottom-right
    #     ((w//2, h-dH), (w//2, h)) # bottom
    # ]
    on = [0] * len(segments)
    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        # cv2.imshow('seg roi', segROI)
        # cv2.waitKey(0)
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        # length = (xB - xA) + (yB - yA)

        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) > 0.50:
            on[i] = 1
        # if total / float(length) >= 0.85:
        #     on[i] = 1

    # lookup the digit and draw it on the image
    digit = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit)

    cv2.rectangle(thresh, (x, y), (x + w, y + h), (200, 100, 255), 2)
    cv2.putText(thresh, str(digit), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

print("digits: {}{}{}".format(*digits[:3]))
cv2.imshow("Output", cv2.resize(thresh, (250, 250)))
cv2.waitKey(0)

#
