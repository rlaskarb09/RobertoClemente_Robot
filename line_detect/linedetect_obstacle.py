import cv2
import numpy as np
# from skimage import measure
# from scipy import ndimage
import time
import logging
import line_detect.line_geom as geom

def find_main_contour(image):
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    C = None
    if cnts is not None and len(cnts) > 0:
        C = max(cnts, key=cv2.contourArea)
    if C is None or cv2.contourArea(C) < 500:
        return cnts, None, None
    rect = cv2.minAreaRect(C)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = geom.order_box(box)
    # cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
    cv2.drawContours(image, C, -1, (0, 0, 255), 3)
    return cnts, C, box
    
def lineDetectObstacle(image, hsv):
    #FIXME
    show = False
    height, width = hsv.shape[:2]

    cx, cy = None, None
    obstacleFlag =False

    # brightness adjust
    mean_val = int(hsv[:,:,2].mean())
    r = 125 / mean_val
    hsv[:,:,2] = np.clip(hsv[:,:,2] * r, 0, 255)

    # saturation adjust
    mean_val = int(hsv[:,:,1].mean())
    r = 125 / mean_val
    hsv[:,:,1] = np.clip(hsv[:,:,1] * r, 0, 255)

    crop = image[2 * int(height / 3): height, :]
    
    obs_roi = hsv
    w, h = width, height / 3
    lower_yellow1 = np.array([20, 80, 100])
    upper_yellow1 = np.array([40, 255, 255])
    # lower_yellow1 = np.array([20, 0, 55])
    # upper_yellow1 = np.array([40, 255, 255])
    lower_yellow2 = np.array([50, 0, 55])
    upper_yellow2 = np.array([75, 255, 255])

    line = cv2.inRange(obs_roi, lower_yellow1, upper_yellow1)
    line = line + cv2.inRange(obs_roi, lower_yellow2, upper_yellow2)
    # max_cont_line, box  = find_main_contour_line(line)
    
    # line center detection
    line_roi = line[2 * int(height / 3): height, :]

    w_offset = 0
    h_offset = int(height - h)

    _, max_cont_line, box = find_main_contour(line_roi)
    thresh1 = cv2.Canny(line_roi, 50,150, apertureSize = 3)
    
    if max_cont_line is not None:
        lines = cv2.HoughLines(thresh1,1, np.pi/180, 200)
        print(len(lines))
        M = cv2.moments(max_cont_line)

        if M['m00'] > 0.0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # cv2.circle(image, (cx + w_offset, cy + h_offset), 2, (255, 0, 0), 2)
            
        else:
            print('No center detected!')
    else:
        print('no line found')

    if show:
        try:
            cv2.imshow('hsv', obs_roi)
            cv2.imshow('crop', crop)
            cv2.imshow('line',line)
        except Exception as e:
            # print(e)
            pass

    return image, cx, cy, obstacleFlag
