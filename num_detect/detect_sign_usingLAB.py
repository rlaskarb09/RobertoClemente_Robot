import cv2
import numpy as np
import time
import os
from skimage import io, color

import num_detect.number_geom as geom
from bright.bright_function import edge_enhancement
from num_detect.detect_function import *

# datafolder = '/Users/soua/Desktop/Project/speed25'
datafolder = '/Users/soua/Desktop/Project/200122nightdemo'
ADDR_LOOKUP = [[1,0,1], [1,0,2], [1,0,3], [2,0,1], [2,0,2], [2,0,3]]

boundaryR = [([1, 20, 200], [20, 150, 255])] # RED
lowerR = np.array(boundaryR[0][0], dtype='uint8')
upperR = np.array(boundaryR[0][1], dtype='uint8')

time_list = []
img_list = []
file_num = len(os.listdir(datafolder))
for imgnum in range(1734):
    imgnum = 636
    imgpath = datafolder + '/Img_%d.jpg'%imgnum
    img_ = cv2.imread(imgpath)
    img = edge_enhancement(img_)
    
    # cv2.imshow('img', img)

    kernel = np.ones((3, 3), np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv)
    # _, S, V = cv2.split(hsv)
    # S =  S + 50
    # hsvR = cv2.merge((_, S, V))
    # cv2.imshow('hsv Red', hsv)
    maskR = cv2.inRange(hsv, lowerR, upperR)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('mask for Red', maskR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    cv2.imshow('A space', A)
    greenA = A.copy()
    greenA[A<90] = 0
    greenA[A > 115] = 0
    cv2.imshow('greenA space', greenA)
    cv2.imshow('B space', B)
    greenB = B.copy()
    greenB[B<140] = 0
    greenB[B > 165] = 0
    maskG = cv2.bitwise_and(greenA, greenA, mask = greenB)

    maskG = cv2.morphologyEx(maskG, cv2.MORPH_OPEN, kernel)
    maskG = cv2.morphologyEx(maskG, cv2.MORPH_CLOSE, kernel)
    maskG = cv2.morphologyEx(maskG, cv2.MORPH_DILATE, kernel)
    cv2.imshow('after B space', B)
    cv2.imshow('mask for green', maskG)
    cv2.waitKey(0)

    start_time = time.time()

    maskR_cnt, maskR_max_cnt, maskR_approx = geom.find_main_contour_approx(maskR)
    try:
        shapeR, thresh = detect(maskR_max_cnt, maskR)
        if shapeR == 'octagon' and cv2.countNonZero(maskR) > 1000:
            cv2.putText(img_, 'STOP', (img_.shape[1] - 100, img_.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    except:
        pass

    maskG_cnt, maskG_max_cnt, maskG_approx = geom.find_main_contour_approx(maskG)
    # cv2.drawContours(img_, maskG_cnt, -1, (0, 255, 0), 2)
    # cv2.imshow('mask Green contour', img_)
    # cv2.waitKey(0)
    W = 0
    H = 0
    # if (maskG_approx is not None) and len(maskG_approx) == 4:
    if maskG_approx is not None and len(maskG_approx) == 4:
        maskG_cnt = max(maskG_cnt, key=cv2.contourArea)
        maskG_rect = cv2.minAreaRect(maskG_cnt)
        maskG_box = np.int0(cv2.boxPoints(maskG_rect))
        W = int(maskG_rect[1][0])
        H = int(maskG_rect[1][1])
    maskG_size = W*H
    # cv2.drawContours(img, maskG_cnt, -1, (0, 0, 255), 1)
    # cv2.imshow('Green contour', img)
    # cv2.waitKey(0)

    # if cv2.countNonZero(maskG) >= 1000 and (maskG_size >= 1000) and (maskG_size < 3000):
    if maskG_size >= 1000 and maskG_size < 3000 :
        peri = cv2.arcLength(maskG_approx, True)
        approx = cv2.approxPolyDP(maskG_approx, 0.02*peri, True)
        warp_img = warp(approx, img)

        # make thresh from warp image
        thresh = make_thresh(warp_img)

        thresh_cut = []
        if thresh.shape[1] >= 45:
            for i in range(3):
                cut_size = thresh.shape[1]//3
                if i == 2:
                    cut_img = thresh[:,i*cut_size:]
                else:
                    cut_img = thresh[:,i*cut_size:(i+1)*cut_size]
                thresh_cut.append(cut_img)

            digits = []
            for i in range(3):
                cut_img = thresh_cut[i]
                each_cnts, max_cnt, box = geom.find_main_contour_box(cut_img.copy())
                (x, y, w, h) = cv2.boundingRect(max_cnt)
                if i == 0:
                    if w < 8:
                        digits.append(1)
                    else:
                        digits.append(2)
                elif i == 1:
                    digits.append(0)
                else:
                    if w < 8 and h >= 10:
                        digits.append(1)
                    elif h >= 10:
                        roi = cut_img[y:y + h, x:x + w]
                        (roiH, roiW) = roi.shape
                        # (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
                        (dH, dW) = (int(roiH * 0.15), int(roiW * 0.2))
                        dHC = int(roiH * 0.1)
                        segments = [
                            ((0, 0), (roiW, dH)),  # top
                            ((0, dH), (dW, roiH // 2)),  # top-left
                            ((roiW - dW, dH), (roiW, roiH // 2)),  # top-right
                            ((0, (roiH // 2) - dHC), (roiW, (roiH // 2) + dHC)),  # center
                            ((0, roiH // 2), (dW, roiH)),  # bottom-left
                            ((roiW - dW, roiH // 2), (roiW, roiH)),  # bottom-right
                            ((0, roiH - dH), (roiW, roiH))  # bottom
                        ]
                        on = [0] * len(segments)
                        for (segi, ((xA, yA), (xB, yB))) in enumerate(segments):
                            segROI = roi[yA:yB, xA:xB]
                            total = cv2.countNonZero(segROI)
                            area = (xB - xA) * (yB - yA)
                            if total / float(area) > 0.55:
                                on[segi] = 1
                        try:
                            digit = DIGITS_LOOKUP[tuple(on)]
                            digits.append(digit)
                        except:
                            pass
            if len(digits) == 3 and digits in ADDR_LOOKUP:
                cv2.drawContours(img_, [maskG_box], -1, (0, 255, 0), 2)
                cv2.putText(img_, '{}{}{}'.format(*digits[:3]), (img_.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            pass
    # cv2.imshow('process...', img_)
    # cv2.waitKey(0)
    time_list.append(time.time()-start_time)
    print('%d th image processing------'%imgnum)
    img_list.append(img_)


print('Avg time for processing : ', sum(time_list)/len(time_list))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('nightdemo.mp4', fourcc, 20.0, (320, 240), True)

for i in range(len(img_list)):
    out.write(img_list[i])

out.release()
cv2.destroyAllWindows()

