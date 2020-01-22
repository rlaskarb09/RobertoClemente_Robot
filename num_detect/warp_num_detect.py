import cv2
import numpy as np
from imutils import contours

from bright.bright_function import edge_enhancement
from num_detect import number_geom as geom


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
img_path = datafolder + '/Img_89.jpg' # 202 s : 56 -> +50 / v : 104 -> 255 - = 151

img = cv2.imread(img_path)
img = cv2.resize(img, (500, 500))
img = edge_enhancement(img)
cv2.imshow('edge enhancement', img)

alpha = 1.5
beta = 30
res = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)
cv2.imshow('scaling img', res)

hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
ab_h, ab_s, ab_v = cv2.split(hsv)
ab_v = 255 - ab_v
ab_s = ab_s + 50
hsv = cv2.merge((ab_h, ab_s, ab_v))
cv2.imshow('alpha beta hsv', hsv)

boundaries = [([60, 90, 80], [90, 255, 110])]  #GREEN
lowerG = np.array(boundaries[0][0], dtype = 'uint8')
upperG = np.array(boundaries[0][1], dtype = 'uint8')

kernel = np.ones((3,3), np.uint8)
mask = cv2.inRange(hsv, lowerG, upperG)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imshow('mask', mask)

output = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow('bitwise output', output)
# cv2.waitKey(0)

cnts, max_cont, box, rect = geom.find_main_contour(mask)
c = max(cnts, key = cv2.contourArea)
img_copy = img.copy()
cv2.drawContours(img_copy, cnts, -1, (255, 0, 0), 2)
cv2.imshow('contour', img_copy)

cv2.drawContours(img, [box], -1, (0, 0, 255), 2)
cv2.imshow('box', img)

our_cnt = None
peri = cv2.arcLength(box, True)
approx = cv2.approxPolyDP(box, 0.02 * peri, True)
if len(approx) == 4:
    our_cnt = approx

warp = warp(our_cnt, img)
cv2.imshow('warp img', warp)

warp_cnt = cv2.findContours(warp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('warp contours', warp_cnt)
cv2.waitKey(0)
shape, th_digits = detect(warp_cnt, ab_v)

digitcnts = []
for cc in box:
    (x, y, w, h) = cv2.boundingRect(cc)
    if (w >= 4 and w <= 40) and (h >= 25 and h<= 30):
        digitcnts.append(cc)

digitcnts = contours.sort_contours(digitcnts, method='left-to-right')[0]
print('%d digits detected'%len(digitcnts))

digits = []
for cc in digitcnts:
    (x, y, w, h) = cv2.boundingRect(cc)
    roi = th_digits[y:y+h, x:x+w]
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)

    cv2.rectangle(th_digits, (x,y), (x+w, y+h), (200, 100, 255), 2)
    cv2.imshow('Output', cv2.resize(th_digits, (250, 250)))
    cv2.waitKey(0)
    print(1)


