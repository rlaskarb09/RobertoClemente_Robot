import cv2
import numpy as np

import line_detect.geom_util as geom
from bright.bright_function import edge_enhancement
import imutils
from imutils import contours
from imutils.perspective import four_point_transform

def detect(c, thresh):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)

        # thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
        kernel = np.ones((4, 4), np.uint8)
        kernel2 = np.ones((5, 5), np.uint8)

        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel2, iterations=1)

        sign = thresh[y:y + h, x:x + w]
        sign = np.rot90(sign, 3)
        thresh = cv2.threshold(sign, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # otherwise, we assume the shape is a circle
    else:
        shape = "octagon"
    return shape, thresh

def warp(cnt, orig):
    pts = cnt.reshape(4,2)
    rect = np.zeros((4, 2), dtype = 'float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (top_l, top_r, bot_r, bot_l) = rect
    widthA = np.sqrt(((bot_r[0] - bot_l[0])**2) + ((bot_r[1] - bot_l[1])**2))
    widthB = np.sqrt(((top_r[0] - top_l[0])**2) + ((top_r[1] - top_l[1])**2))

    heightA = np.sqrt(((top_r[0] - bot_r[0])**2) + ((top_r[1] - bot_r[1])**2))
    heightB = np.sqrt(((top_l[0] - bot_l[0])**2) + ((top_l[1] - bot_l[1])**2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0,0], [maxWidth-1,0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    return warp

datafolder = '/Users/soua/Desktop/Project/sterling_demo'
imgpath = datafolder + '/Img_1077.jpg'

img = cv2.imread(imgpath)
img = edge_enhancement(img)

alpha = 1.5
beta = 50
res = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v = 255 - v
s = s + 50
hsv = cv2.merge((h, s, v))
cv2.imshow('alpha beta hsv', hsv)

# boundaries = [([60, 90, 80], [90, 255, 110])]  #GREEN
boundaries = [([50, 80, 0], [90, 150, 20])]
lowerG = np.array(boundaries[0][0], dtype = 'uint8')
upperG = np.array(boundaries[0][1], dtype = 'uint8')

kernel = np.ones((3,3), np.uint8)
mask = cv2.inRange(hsv, lowerG, upperG)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

img_copy = img.copy()
cnts, max_cont, approx_cnt = geom.find_main_contour(mask)
cv2.drawContours(img_copy, max_cont, -1, (255, 0, 0), 2)
cv2.imshow('first contour', img_copy)
cv2.drawContours(img_copy, [approx_cnt], -1, (0, 0, 255), 2)
cv2.imshow('approx contour', img_copy)

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
cv2.imshow('warp gray', warp_gray)

# warp_gray= edge_enhancement(warp_gray)
thresh_ = cv2.threshold(warp_gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.waitKey(0)
thresh = 255 - thresh_
thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
cv2.imshow('thresh', thresh)
# cv2.waitKey(0)

thresh_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
thresh_cnts = imutils.grab_contours(thresh_cnts)

digitcnts = []
for c in thresh_cnts:
    (x, y, w, h_) = cv2.boundingRect(c)
    if w >= 15 and (h_ >= 20 and h_ <= 40):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        digitcnts.append(box)
    elif w < 10 and (h_ >= 20 and h_ <= 40):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        digitcnts.append(box)

cv2.drawContours(warp, digitcnts, -1, (255, 0, 0), 2)
cv2.imshow('draw digit', warp)
cv2.waitKey(0)

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    # (1, 1, 1, 1, 1, 1, 1): 1,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3
}

digits = []
for cc in digitcnts:
    (x, y, w, h) = cv2.boundingRect(cc)
    if w < 10:
        w = w+8
        x = x-8
    roi = thresh[y:y+h, x:x+w]
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.3), int(roiH * 0.2))
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

        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) >= 0.6:
            on[i] = 1

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
