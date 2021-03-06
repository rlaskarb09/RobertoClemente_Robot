import cv2
import numpy as np
import time
import os

import num_detect.number_geom as geom
from bright.bright_function import edge_enhancement
from num_detect.detect_function import *

# datafolder = '/Users/soua/Desktop/Project/speed25'
datafolder = '/Users/soua/Desktop/Project/200122nightdemo'
ADDR_LOOKUP = [[1,0,1], [1,0,2], [1,0,3], [2,0,1], [2,0,2], [2,0,3]]

time_list = []
img_list = []
file_num = len(os.listdir(datafolder))
for imgnum in range(file_num):
    imgnum = 2
    imgpath = datafolder + '/Img_%d.jpg'%imgnum
    img_ = cv2.imread(imgpath)
    img = edge_enhancement(img_)
    res = cv2.convertScaleAbs(img_, alpha=1.5, beta=70)

    equ = cv2.equalizeHist(cv2.cvtColor(res.copy(), cv2.COLOR_BGR2GRAY))
    cv2.imshow('equ', equ)
    equ[equ < 190] = 0
    equ[equ >= 190] = 255
    equ_img = cv2.bitwise_and(img_, img_, mask = equ)
    cv2.imshow('and equ', equ_img)
    # cv2.waitKey(0)
    start_time = time.time()

    # make hsv for Green and Red
    # img = edge_enhancement(img_)
    # res = cv2.convertScaleAbs(img_, alpha=1.5, beta=50)

    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    cv2.imshow('original hsv', hsv)
    H, S, V = cv2.split(hsv)
    S = S + 50
    hsvR = cv2.merge((H, S, V))
    V = 255 - V # red 조절 (엎으면 아예 red 없음)
    hsvG = cv2.merge((H, S, V))
    cv2.imshow('hsvG', hsvG)
    # cv2.waitKey(0)
    # make mask for Green and Red
    maskG, maskR = make_mask(hsvG, hsvR)
    cv2.imshow('Green Mask', maskG)
    # cv2.waitKey(0)

    maskG_equ = cv2.bitwise_and(equ, equ, mask = maskG)
    cv2.imshow('equ and mask', maskG_equ)
    cv2.waitKey(0)
    # check STOP sign is in frame
    stop_img_ = check_STOP(img_, maskR)

    # check Green mask size in frame
    # maskG_approx, maskG_size = check_green_size(maskG)
    maskG_cnt, maskG_max_cnt, maskG_approx = geom.find_main_contour_approx(maskG_equ)
    # cv2.drawContours(img_, maskG_cnt, -1, (0, 255, 0), 2)
    # cv2.imshow('mask Green contour', img_)
    # cv2.waitKey(0)
    W = 0
    H = 0
    if (maskG_approx is not None) and len(maskG_approx) == 4:
        maskG_cnt = max(maskG_cnt, key=cv2.contourArea)
        maskG_rect = cv2.minAreaRect(maskG_cnt)
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
                        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
                        dHC = int(roiH * 0.1)
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
                        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                            segROI = roi[yA:yB, xA:xB]
                            total = cv2.countNonZero(segROI)
                            area = (xB - xA) * (yB - yA)
                            if total / float(area) > 0.50:
                                on[i] = 1
                        try:
                            digit = DIGITS_LOOKUP[tuple(on)]
                            digits.append(digit)
                        except:
                            pass
            if len(digits) == 3 and digits in ADDR_LOOKUP:
                cv2.putText(stop_img_, '{}{}{}'.format(*digits[:3]), (img_.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            pass
    # cv2.imshow('process...', stop_img_)
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

