import cv2
import numpy as np
from bright.bright_function import edge_enhancement
from num_detect.detect_function import detect, warp
import num_detect.number_geom as geom

# boundaryR = [([5, 5, 255], [20, 100, 255])] # RED
boundaryR = [([1, 5, 200], [20, 120, 255])] # RED
lowerR = np.array(boundaryR[0][0], dtype='uint8')
upperR = np.array(boundaryR[0][1], dtype='uint8')

def detect_stop_sign(image):
    flag = None
    lowerR = np.array(boundaryR[0][0], dtype='uint8')
    upperR = np.array(boundaryR[0][1], dtype='uint8')
    img = edge_enhancement(image)
    alpha = 1.5
    beta = 50
    res = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    S = S + 50
    hsvR = cv2.merge((H, S, V))
    kernel = np.ones((3, 3), np.uint8)
    maskR = cv2.inRange(hsvR, lowerR, upperR)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_CLOSE, kernel)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_OPEN, kernel)
    cntR, max_cntR, approx_cntR = geom.find_main_contour_approx(maskR)
    try:
        shapeR, thresh = detect(max_cntR, maskR)
        if shapeR == 'octagon' and cv2.countNonZero(maskR) > 1000:
            cv2.putText(hsv, 'STOP', (image.shape[1]-100, image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            flag = True

    except:
        pass
    return hsv, flag