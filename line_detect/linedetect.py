import cv2
import numpy as np
# from skimage import measure
# from scipy import ndimage
import time
import logging
import line_detect.line_geom as geom

def find_main_contour(image):
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    C = None
    if cnts is not None and len(cnts) > 0:
        C = max(cnts, key=cv2.contourArea)

    if C is None:
        return cnts, None, None

    rect = cv2.minAreaRect(C)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = geom.order_box(box)
    # cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
    # cv2.drawContours(image, C, -1, (0, 0, 255), 3)
    # cv2.imshow('contour' , image)
    return cnts, C, box

def adjust_brightness(img, level):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b = np.mean(hsv[:,:,2])
    if b == 0:
        return hsv
    r = level / b
    c = hsv.copy()
    c[:,:,2] = c[:,:,2] * r
    return c

def prepare_pic(hsv):
    height, width = hsv.shape[:2]
    crop = hsv[2 * int(height / 3): height, :]
    # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # # hsv2 = adjust_brightness(crop, 100)
    # # cv2.imshow('hsv', hsv)
    # # cv2.imshow('hs2', hsv2)

    lower_yellow = np.array([20, 30, 150])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(crop, lower_yellow, upper_yellow)

    # cv2.imshow('mask', mask)

    return mask, width, height / 3

# def prepare_pic2(image):
#     height, width = image.shape[:2]
#     crop = image[2 * int(height / 3): height, :]
#
#     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#     # hsv2 = adjust_brightness(crop, 100)
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#
#     lower_yellow = np.array([20, 80, 100])
#     upper_yellow = np.array([40, 255, 255])
#     ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#     crop_binary = cv2.bitwise_and(crop, crop, mask=thresh1)
#     mask = cv2.inRange(crop_binary, lower_yellow, upper_yellow)
#     # adap_mean_2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)
#     # cv2.imshow('mask', mask)
#     return mask, width, height / 3

# def edge_enhancement(image):
#     kernel = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]])/8.0
#     output = cv2.filter2D(image, -1, kernel)
#     return output

def lineDetect(image, hsv):
    obstacleFlag = False
    height, width = hsv.shape[:2]
    mask, w, h = prepare_pic(hsv)
    # markers = ndimage.label(mask, structure=np.ones((3, 3)))[0]
    # labels = watershed(-D, markers, mask=thresh)
    # print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    conts, max_cont, box = find_main_contour(mask)

    angle = None
    shift = None
    if box is not None:
        # labels = measure.label(mask, connectivity=2, background=0)
        # segmentNum = 0
        # for label in np.unique(labels):
        #     # if this is the background label, ignore it
        #     if label == 0:
        #         continue
        #
        #     # otherwise, construct the label mask and count the
        #     # number of pixels
        #     labelMask = np.zeros(mask.shape, dtype="uint8")
        #     labelMask[labels == label] = 255
        #     numPixels = cv2.countNonZero(labelMask)
        #
        #     # if the number of pixels in the component is sufficiently
        #     # large, then add it to our mask of "large blobs"
        #     if numPixels > 300:
        #         # mask = cv2.add(mask, labelMask)
        #         segmentNum += 1
        #
        # if segmentNum >= 2:
        #     obstacleFlag = True
        p1, p2 = geom.calc_box_vector(box)

        if p1 is not None:
            angle = geom.get_vert_angle(p1, p2, w, h)
            shift = geom.get_horz_shift(p1[0], w)

            msg_a = "Angle {0}".format(int(angle))
            msg_s = "Shift {0}".format(shift)

            w_offset = int((width - w) / 2)
            h_offset = int(height - h)

            dbox = geom.shift_box(box, w_offset, h_offset)

            dp1 = (int(p1[0] + w_offset), int(p1[1] + h_offset))
            dp2 = (int(p2[0] + w_offset), int(p2[1] + h_offset))
            # cv2.drawContours(cropped, max_cont, -1, (0, 255, 0), 3)
            # cv2.drawContours(image, max_cont, -1, (0, 0, 255), 3)

            cv2.line(image, dp1, dp2, (255, 0, 0), 3)
            cv2.putText(image, msg_a, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(image, msg_s, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.drawContours(image, [dbox], -1, (0, 255, 0), 3)

    return image, shift, angle, obstacleFlag

# def lineDetectObstacle(image, hsv):
#     obstacleFlag =False
#     height, width = hsv.shape[:2]
#     mask, cropped, w, h = prepare_pic(hsv)
#     cv2.imshow('mask', mask)
#     time.time()
#
#     # markers = ndimage.label(mask, structure=np.ones((3, 3)))[0]
#     # labels = watershed(-D, markers, mask=thresh)
#     # print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
#     crop_height = 2*int(height/3)
#
#     conts, max_cont, _ = find_main_contour(mask)
#     _, _, box = find_main_contour(cropped)
#
#     angle = None
#     shift = None
#     if box is not None:
#         labels = measure.label(mask, connectivity=2, background=0)
#         segmentNum = 0
#         for label in np.unique(labels):
#             # if this is the background label, ignore it
#             if label == 0:
#                 continue
#
#             # otherwise, construct the label mask and count the
#             # number of pixels
#             labelMask = np.zeros(mask.shape, dtype="uint8")
#             labelMask[labels == label] = 255
#             numPixels = cv2.countNonZero(labelMask)
#
#             # if the number of pixels in the component is sufficiently
#             # large, then add it to our mask of "large blobs"
#             if numPixels > 300:
#                 # mask = cv2.add(mask, labelMask)
#                 segmentNum += 1
#
#         if segmentNum >= 2:
#             obstacleFlag = True
#
#         p1, p2 = geom.calc_box_vector(box)
#
#         if p1 is not None:
#             angle = geom.get_vert_angle(p1, p2, w, h)
#             shift = geom.get_horz_shift(p1[0], w)
#
#             msg_a = "Angle {0}".format(int(angle))
#             msg_s = "Shift {0}".format(shift)
#
#             w_offset = int((width - w) / 2)
#             h_offset = int(height - h)
#
#             dbox = geom.shift_box(box, w_offset, h_offset)
#
#             dp1 = (int(p1[0] + w_offset), int(p1[1] + h_offset))
#             dp2 = (int(p2[0] + w_offset), int(p2[1] + h_offset))
#             # cv2.drawContours(cropped, max_cont, -1, (0, 255, 0), 3)
#             cv2.drawContours(image, max_cont, -1, (0, 0, 255), 3)
#
#             cv2.line(image, dp1, dp2, (255, 0, 0), 3)
#             cv2.putText(image, msg_a, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#             cv2.putText(image, msg_s, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#             cv2.drawContours(image, [dbox], -1, (0, 255, 0), 3)
#
#     return image, shift, angle, obstacleFlag
