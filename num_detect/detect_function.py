import cv2
import numpy as np

from bright.bright_function import edge_enhancement
import num_detect.number_geom as geom

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
     (1, 1, 1, 1, 1, 1, 1): 0,
    # (1, 1, 1, 1, 1, 1, 1): 1,
    (0, 0, 1, 0, 0, 1, 0): 1,

    (1, 0, 1, 1, 1, 0, 1): 2,
    (0, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 1, 0, 0): 2,

    (1, 0, 1, 1, 0, 1, 1): 3,
    (1, 0, 1, 1, 0, 0, 0): 3,
    (1, 0, 1, 1, 0, 0, 1): 3,
    (1, 0, 1, 0, 0, 1, 1): 3,
    (1, 0, 0, 1, 0, 1, 1): 3,

}


def make_mask(hsvG, hsvR, lowerG, upperG, lowerR, upperR):
    kernel = np.ones((3, 3), np.uint8)
    maskG = cv2.inRange(hsvG, lowerG, upperG)
    # maskG = cv2.inRange(hsvG, np.uint8([36, 54, 100]), np.uint8([76, 74, 140]))
    maskG = cv2.morphologyEx(maskG, cv2.MORPH_CLOSE, kernel)

    maskR = cv2.inRange(hsvR, lowerR, upperR)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_CLOSE, kernel)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_OPEN, kernel)
    return maskG, maskR

def check_STOP(image, maskR):
    cntR, max_cntR, approx_cntR = geom.find_main_contour_approx(maskR)
    if max_cntR is not None:
    # try:
        shapeR, thresh = detect(max_cntR, maskR)
        if shapeR == 'octagon' and cv2.countNonZero(maskR) > 1000:
            cv2.putText(image, 'STOP', (image.shape[1] - 100, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # except:
    else:
        pass
    return image

def check_green_size(maskG):
    maskG_cnt, maskG_max_cnt, maskG_approx = geom.find_main_contour_approx(maskG)
    W = 0
    H = 0
    if (maskG_approx is not None) and len(maskG_approx) == 4:
        maskG_cnt = max(maskG_cnt, key=cv2.contourArea)
        maskG_rect = cv2.minAreaRect(maskG_cnt)
        W = int(maskG_rect[1][0])
        H = int(maskG_rect[1][1])
    maskG_size = W*H
    return maskG_approx, maskG_size

def make_thresh(warp_img):
    warp_gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
    warp_gray = cv2.GaussianBlur(warp_gray, (3, 3), 0)
    thresh = cv2.threshold(warp_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    th_kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, th_kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, th_kernel)
    return thresh

def detect(c, thresh):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01*peri, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
        thresh_s = thresh

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
        thresh_s = cv2.threshold(sign, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # otherwise, we assume the shape is a circle
    elif len(approx) > 4 :
        shape = "octagon"
        thresh_s = thresh
    else:
        shape = "OTHERS"
        thresh_s = thresh
    return shape, thresh_s

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