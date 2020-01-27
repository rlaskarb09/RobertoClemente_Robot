import cv2
import numpy as np
from bright.bright_function import edge_enhancement
from num_detect.detect_function import detect, warp
import num_detect.number_geom as geom

# boundaryG = [([50,80,0], [90,255,100])] # sterling_demo2 102
boundaryG = [([60, 70, 0], [90, 255, 90])]
lowerG = np.array(boundaryG[0][0], dtype = 'uint8')
upperG = np.array(boundaryG[0][1], dtype = 'uint8')

# boundaryR = [([5, 5, 255], [20, 100, 255])] # RED
boundaryR = [([5, 5, 250], [20, 120, 255])] # RED
lowerR = np.array(boundaryR[0][0], dtype='uint8')
upperR = np.array(boundaryR[0][1], dtype='uint8')

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
    (1, 0, 1, 1, 0, 0, 1): 3
}

def prepare_pic(image):
    img = edge_enhancement(image)

    alpha = 1.5
    beta = 50
    res = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)

    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1] + 50
    # H, S, V = cv2.split(hsv)
    # S = S + 50
    # hsvR = cv2.merge((H, S, V))
    return hsv

def stopDetect(image, hsv):
    stop_detected = False

    #Stop detection
    lowerR = np.array(boundaryR[0][0], dtype='uint8')
    upperR = np.array(boundaryR[0][1], dtype='uint8')

    kernel = np.ones((3, 3), np.uint8)
    maskR = cv2.inRange(hsv, lowerR, upperR)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_CLOSE, kernel)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_OPEN, kernel)
    cntR, max_cntR, approx_cntR = geom.find_main_contour_approx(maskR)

    try:
        shapeR, thresh = detect(max_cntR, maskR)
        if shapeR == 'octagon' and cv2.countNonZero(maskR) > 1000:
            cv2.putText(image, 'STOP', (image.shape[1]-100, image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            stop_detected = True
    except:
        pass
    return image, stop_detected

def addressDetect(img, hsv):

    address = None

    hsv[:,:,2] = 255 - hsv[:,:,2]

    #Address detection
    lowerG = np.array(boundaryG[0][0], dtype='uint8')
    upperG = np.array(boundaryG[0][1], dtype='uint8')
    maskG = cv2.inRange(hsv, lowerG, upperG)

    kernel = np.ones((3, 3), np.uint8)
    maskG = cv2.morphologyEx(maskG, cv2.MORPH_CLOSE, kernel)
    maskG_cnt, _, maskG_approx = geom.find_main_contour_approx(maskG)

    W = 0
    H = 0
    if (maskG_approx is not None) and len(maskG_approx) == 4:
        maskG_cnt = max(maskG_cnt, key=cv2.contourArea)
        maskG_rect= cv2.minAreaRect(maskG_cnt)
        W = int(maskG_rect[1][0])
        H = int(maskG_rect[1][1])

    if cv2.countNonZero(maskG) >= 1000 and (W*H >= 1000) and (W*H < 3000):
        cntG, max_cntG, approx_cntG = geom.find_main_contour_approx(maskG)
        peri = cv2.arcLength(approx_cntG, True)
        approx = cv2.approxPolyDP(approx_cntG, 0.02*peri, True)
        if len(approx) == 4 :
            our_cnt = approx
        warp_img = warp(our_cnt, img)

        warp_gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
        warp_gray = cv2.GaussianBlur(warp_gray, (3, 3), 0)
        thresh = cv2.threshold(warp_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        th_kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, th_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, th_kernel)

        thresh_cut = []
        if thresh.shape[1] < 50:
            pass
        else:
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
            if len(digits) == 3:
                address = '{}{}{}'.format(*digits[:3])
                cv2.putText(img, '{}{}{}'.format(*digits[:3]), (img.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img, address
