import cv2
import numpy as np
from bright.bright_function import edge_enhancement, adjust_gamma
from num_detect.detect_function import detect, warp, make_thresh
import num_detect.number_geom as geom

# boundaryG = [([50,80,0], [90,255,100])] # sterling_demo2 102
ADDR_LOOKUP = [[1,0,1], [1,0,2], [1,0,3], [2,0,1], [2,0,2], [2,0,3]]

boundaryR = [([1, 20, 200], [20, 150, 255])] # RED
lowerR = np.array(boundaryR[0][0], dtype='uint8')
upperR = np.array(boundaryR[0][1], dtype='uint8')

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
     (1, 1, 1, 1, 1, 1, 1): 0,

    (1, 1, 1, 1, 1, 1, 1): 1,
    (0, 0, 1, 0, 0, 1, 0): 1,

    (1, 0, 1, 1, 1, 0, 1): 2,
    (0, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 1, 0, 0): 2,

    (1, 0, 1, 1, 0, 1, 1): 3,
    (1, 0, 1, 1, 0, 0, 0): 3,
    (1, 0, 1, 1, 0, 0, 1): 3,
    (1, 0, 1, 0, 0, 1, 1): 3,
    (1, 0, 0, 1, 0, 1, 1): 3,
    (1, 0, 1, 1, 0, 1, 0): 3,
}

def prepare_stop_pic(image):
    img = edge_enhancement(image)

    # alpha = 1.5
    # beta = 50
    # res = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv[:,:,1] = hsv[:,:,1] + 50
    # H, S, V = cv2.split(hsv)
    # S = S + 50
    # hsvR = cv2.merge((H, S, V))
    return hsv

def prepare_addr_pic(image):
    img = edge_enhancement(image)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    if np.mean(L) < 120:
        img = adjust_gamma(img, gamma=0.6)
        L, A, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

    greenA = A.copy()
    greenA[A < 90] = 0
    greenA[A > 115] = 0

    greenB = B.copy()
    greenB[B < 140] = 0
    greenB[B > 160] = 0

    maskG = cv2.bitwise_and(greenA, greenA, mask=greenB)
    kernel = np.ones((3, 3), np.uint8)
    maskG = cv2.morphologyEx(maskG, cv2.MORPH_OPEN, kernel)
    maskG = cv2.morphologyEx(maskG, cv2.MORPH_CLOSE, kernel)
    maskG = cv2.morphologyEx(maskG, cv2.MORPH_DILATE, kernel)
    return maskG

def stopDetect(image, hsv):
    stop_detected = False

    #Stop detection
    # lowerR = np.array(boundaryR[0][0], dtype='uint8')
    # upperR = np.array(boundaryR[0][1], dtype='uint8')

    kernel = np.ones((3, 3), np.uint8)
    maskR = cv2.inRange(hsv, lowerR, upperR)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_CLOSE, kernel)
    # maskR = cv2.morphologyEx(maskR, cv2.MORPH_OPEN, kernel)

    maskR_cnt, maskR_max_cnt, maskR_approx = geom.find_main_contour_approx(maskR)
    try:
        shapeR, thresh = detect(maskR_max_cnt, maskR)
        if shapeR == 'octagon' and cv2.countNonZero(maskR) > 1000:
            cv2.putText(image, 'STOP', (image.shape[1] - 100, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            stop_detected = True
    except:
        pass

    return image, stop_detected

def addressDetect(img, maskG):

    address = None

    maskG_cnt, _, maskG_approx = geom.find_main_contour_approx(maskG)
    W = 0
    H = 0
    if maskG_approx is not None and len(maskG_approx) == 4:
        maskG_cnt = max(maskG_cnt, key=cv2.contourArea)
        maskG_rect = cv2.minAreaRect(maskG_cnt)
        maskG_box = np.int0(cv2.boxPoints(maskG_rect))
        W = int(maskG_rect[1][0])
        H = int(maskG_rect[1][1])
    maskG_size = W * H

    if maskG_size >= 1000 and maskG_size < 3000:
        peri = cv2.arcLength(maskG_approx, True)
        approx = cv2.approxPolyDP(maskG_approx, 0.02 * peri, True)
        warp_img = warp(approx, img)

        thresh = make_thresh(warp_img)

        thresh_cut = []
        if thresh.shape[1] >= 45:
            for i in range(3):
                cut_size = thresh.shape[1] // 3
                if i == 2:
                    cut_img = thresh[:, i * cut_size:]
                else:
                    cut_img = thresh[:, i * cut_size:(i + 1) * cut_size]
                thresh_cut.append(cut_img)

            digits = []
            for i in range(3):
                cut_img = thresh_cut[i]
                each_cnts, max_cnt, box = geom.find_main_contour_box(cut_img.copy())
                (x, y, w, h) = cv2.boundingRect(max_cnt)
                # (x, y, w, h) = box
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
            if len(digits) == 3:
                address = '{}{}{}'.format(*digits[:3])
                cv2.drawContours(img, [maskG_box], -1, (0, 255, 0), 2)
                cv2.putText(img, '{}{}{}'.format(*digits[:3]), (img.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img, address
