import cv2
import numpy as np
# from skimage import measure
# from scipy import ndimage
import time
import logging
import line_detect.line_geom as geom

def find_main_contour_obs(image):
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    C = None
    if cnts is not None and len(cnts) > 0:
        C = max(cnts, key=cv2.contourArea)
    if C is None or cv2.contourArea(C) < 2500:
        return cnts, None, None

    rect = cv2.minAreaRect(C)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = geom.order_box(box)
    cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
    cv2.drawContours(image, C, -1, (0, 0, 255), 3)
    return cnts, C, box

def find_main_contour(image):
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    C = None
    if cnts is not None and len(cnts) > 0:
        C = max(cnts, key=cv2.contourArea)
        # print(cv2.contourArea(C))
    if C is None or cv2.contourArea(C) < 30:
        return cnts, None, None
    rect = cv2.minAreaRect(C)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = geom.order_box(box)
    # cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
    cv2.drawContours(image, C, -1, (0, 0, 255), 3)
    return cnts, C, box

def find_main_contour_line(image):
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        box = None
        C = None
        cnts_line = np.array([])
        for cnt in cnts:
            # print(cv2.contourArea(cnt))
            if cv2.contourArea(cnt) < 30:
                break
            else:
                cnts_line = np.vstack([cnts_line, cnt]) if cnts_line.size else cnt
        if len(cnts_line) > 0 :
            rect = cv2.minAreaRect(cnts_line)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = geom.order_box(box)
            return cnts_line, box
        else:
            return None, None
    else:
        return None, None


# def find_main_contour_obs(image, lineLeft, lineRight, lineTop, lineBot):
#     cnts, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if len(cnts) > 0:
#         cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#         box = None
#         C = None
#         cnts_obs = np.array([])
#         for cnt in cnts:
#             # print(cv2.contourArea(cnt))
#             if cv2.contourArea(cnt) < 100:
#                 break
#             else:
#                 obsLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
#                 obsRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
#                 obsTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
#                 obsBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
#                 if not((obsRight < lineLeft) or (obsLeft > lineRight) or ((obsLeft > lineLeft) and (obsRight < lineRight) )):
#                     cnts_obs = np.vstack([cnts_obs, cnt]) if cnts_obs.size else cnt
#         if len(cnts_obs) > 0 :
#             rect = cv2.minAreaRect(cnts_obs)
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
#             box = geom.order_box(box)
#             return cnts_obs, box
#         else:
#             return None, None
#     else:
#         return None, None

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
#
# def lineDetect(image, hsv):
#     obstacleFlag = False
#     height, width = hsv.shape[:2]
#     mask, w, h = prepare_pic(hsv)
#     # markers = ndimage.label(mask, structure=np.ones((3, 3)))[0]
#     # labels = watershed(-D, markers, mask=thresh)
#     # print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
#     conts, max_cont, box = find_main_contour(mask)
#
#     angle = None
#     shift = None
#     if box is not None:
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
#             # cv2.drawContours(image, max_cont, -1, (0, 0, 255), 3)
#
#             cv2.line(image, dp1, dp2, (255, 0, 0), 3)
#             cv2.putText(image, msg_a, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#             cv2.putText(image, msg_s, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#             cv2.drawContours(image, [dbox], -1, (0, 255, 0), 3)
#
#     return image, shift, angle, obstacleFlag

# def lineDetectObstacle(frame, hsv, last_cx):
#     show=False
#     cx, cy = None, None
#     obstacleFlag = False
#     height, width = hsv.shape[:2]
#
#     w, h = width, height / 3
#
#     # brightness adjust
#     mean_val = int(hsv[:,:,2].mean())
#     r = 125 / mean_val
#     hsv[:,:,2] = np.clip(hsv[:,:,2] * r, 0, 255)
#
#     # saturation adjust
#     mean_val = int(hsv[:,:,1].mean())
#     r = 125 / mean_val
#     hsv[:,:,1] = np.clip(hsv[:,:,1] * r, 0, 255)
#     crop = hsv[2 * int(height / 3): height, :]
#
#     #FIXME
#     # line
#     lower_yellow1 = np.array([20, 20, 160])
#     upper_yellow1 = np.array([40, 255, 255])
#     lower_yellow2 = np.array([50, 0, 150])
#     upper_yellow2 = np.array([85, 20, 255])
#     line = cv2.inRange(crop, lower_yellow1, upper_yellow1)
#     line = line + cv2.inRange(crop, lower_yellow2, upper_yellow2)
#     # kernel = np.ones((5, 5), np.uint8)
#     # line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel)
#
#     # mask = cv2.inRange(crop, lower_yellow, upper_yellow)
#     # mask_black = cv2.inRange(crop, lower_black, upper_black)
#     # mask_obstacle = 255 - cv2.bitwise_or(mask, mask_black)
#     cnts, hierarchy = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if cnts is not None and len(cnts) > 0:
#         max_cont_line = max(cnts, key=cv2.contourArea)
#         # max_cont_line, box = find_main_contour_line(mask)
#         if max_cont_line is not None:   #line detected
#             #
#             # # determine the most extreme points along the contour
#             # lineLeft = tuple(max_cont_line[max_cont_line[:, :, 0].argmin()][0])
#             # lineRight = tuple(max_cont_line[max_cont_line[:, :, 0].argmax()][0])
#             # lineTop = tuple(max_cont_line[max_cont_line[:, :, 1].argmin()][0])
#             # lineBot = tuple(max_cont_line[max_cont_line[:, :, 1].argmax()][0])
#             #
#             # max_cont_obs, box_obs = find_main_contour_obs(mask_obstacle, lineLeft, lineRight, lineTop, lineBot)
#             #
#             # if max_cont_obs is not None:
#             #     print("obstacle detected")
#             #     obstacleFlag = True
#
#             M = cv2.moments(max_cont_line)
#             if M['m00'] > 0.0:
#                 cx = int(M['m10'] / M['m00'])
#                 cy = int(M['m01'] / M['m00'])
#             else:
#                 cx = last_cx
#                 print('No center detected!')
#                 # crop_height, crop_width = crop.shape[:2]
#                 # cx, cy = int(crop_width / 2), int(crop_height / 2)
#
#     else:       #line not detected
#     #     max_cont_obs, box_obs = find_main_contour_obs(mask_obstacle, (0,int(h/2)), (w,int(h/2)), (int(w/2),0), (int(w/2),h))
#     #     if max_cont_obs is not None:
#     #         print("obstacle detected")
#     #         obstacleFlag = True
#     #     crop_height, crop_width = crop.shape[:2]
#         print('no line detected!')
#         # cx, cy = int(crop_width / 2), int(crop_height / 2)
#     try:
#         w_offset = 0
#         h_offset = int(height - h)
#         cv2.circle(frame, (cx+w_offset, cy+h_offset), 2, (0, 0, 255), 2)
#         # cv2.drawContours(frame, (cx, cy), 2, (255, 0, 0), 2)
#         # print('cx, cy =', cx, cy)
#     except:
#         print('exception in cv2.circle. cx, cy =', cx, cy)
#
#     try:
#         w_offset = 0
#         h_offset = int(height -h)
#         # dbox = geom.shift_box(box_obs, w_offset, h_offset)
#         # cv2.drawContours(frame, [dbox], -1, (0, 125, 125), 2)
#     except:
#         print('no obstacle')
#
#     if show:
#         try:
#             cv2.imshow('hsv', hsv)
#             cv2.imshow('line', line)
#             # cv2.imshow('floor', floor)
#             # cv2.imshow('obs', obstacle)
#         except Exception as e:
#             print(e)
#             pass
#     return frame, cx, cy, obstacleFlag
import pdb

def lineDetectObstacle(image, hsv):
    #FIXME
    show = True
    height, width = hsv.shape[:2]

    cx, cy = None, None
    obstacleFlag =False

    # # brightness adjust
    # mean_val = int(hsv[:,:,2].mean())
    # r = 125 / mean_val
    # hsv[:,:,2] = np.clip(hsv[:,:,2] * r, 0, 255)
    #
    # # saturation adjust
    # mean_val = int(hsv[:,:,1].mean())
    # r = 125 / mean_val
    # hsv[:,:,1] = np.clip(hsv[:,:,1] * r, 0, 255)

    # crop = image[2 * int(height / 3): height, :]
    w, h = width, height / 15

    lower_yellow1 = np.array([20, 0, 170])
    upper_yellow1 = np.array([40, 255, 255])
    lower_yellow2 = np.array([40, 0, 150])
    upper_yellow2 = np.array([85, 20, 255])

    mask = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
    # mask = mask + cv2.inRange(hsv, lower_yellow2, upper_yellow2)

    # line center detection
    line_roi = mask[14 * int(height / 15): height, :]
    w_offset = 0
    h_offset = int(height - h)

    cnt1, max_cont_line, box = find_main_contour(line_roi)

    if max_cont_line is not None:
        # black_img = np.zeros((line_roi.shape))
        # cv2.drawContours(black_img, max_cont_line, -1, (255, 0, 0), 2)
        # thresh1 = cv2.threshold(black_img, 0, 255, cv2.THRESH_BINARY)
        # import pdb
        # pdb.set_trace()
        # lines = cv2.HoughLines(thresh1,1, np.pi/180, 200)
        # print(len(lines))
        M = cv2.moments(max_cont_line)

        if M['m00'] > 0.0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(image, (cx + w_offset, cy + h_offset), 2, (255, 0, 0), 2)

        else:
            print('No center detected!')
    else:
        obstacleFlag = True
    # obstacle detection
    # obs_roi = mask[1 * int(height / 3):  2* int(height / 3), :]
    # cnt2, max_cont_obs, box = find_main_contour(obs_roi)
    #
    # if max_cont_obs is None:
    #     obstacleFlag = True
    #     msg = "Obstacle"
    #     cv2.putText(image, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # # black_img = np.zeros((line_roi.shape))
        # # cv2.drawContours(black_img, max_cont_line, -1, (255, 0, 0), 2)
        # # thresh1 = cv2.threshold(black_img, 0, 255, cv2.THRESH_BINARY)
        # # import pdb
        # # pdb.set_trace()
        # # lines = cv2.HoughLines(thresh1,1, np.pi/180, 200)
        # # print(len(lines))
        # M = cv2.moments(max_cont_line)
        #
        # if M['m00'] > 0.0:
        #     cx = int(M['m10'] / M['m00'])
        #     cy = int(M['m01'] / M['m00'])
        #     cv2.circle(image, (cx + w_offset, cy + h_offset), 2, (255, 255, 0), 2)
        #
        # else:
        #     print('No center detected!')

    if show:
        try:
            cv2.imshow('hsv', hsv)
            # cv2.imshow('crop', crop)
            cv2.imshow('mask', mask)
            cv2.imshow('image', image)

        except Exception as e:
            print(e)
        #     pass

    return image, cx, cy, obstacleFlag

def lineLength(p1, p2):
    return (np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

# def lineDetectObstacle(image, hsv):
#     #FIXME
#     show = False
#     height, width = hsv.shape[:2]
#     cx, cy = None, None
#     obstacleFlag =False
#
#     # brightness adjust
#     mean_val = int(hsv[:,:,2].mean())
#     r = 125 / mean_val
#     hsv[:,:,2] = np.clip(hsv[:,:,2] * r, 0, 255)
#
#     # saturation adjust
#     mean_val = int(hsv[:,:,1].mean())
#     r = 125 / mean_val
#     hsv[:,:,1] = np.clip(hsv[:,:,1] * r, 0, 255)
#     # # print(r)
#     # hsv[:,:,2] = hsv[:,:,2] * r
#
#     obs_roi = hsv[1 * int(height / 3): height, :]
#     w, h = width, height / 3
#     w_offset = int(width - w)
#     h_offset = int(height - h)
#
#     #FIXME
#     # line
#     lower_yellow1 = np.array([20, 20, 160])
#     upper_yellow1 = np.array([40, 255, 255])
#     lower_yellow2 = np.array([50, 0, 150])
#     upper_yellow2 = np.array([85, 20, 255])
#
#     #FIXME
#     # floor
#     lower_black1 = np.array([0, 0, 0])
#     upper_black1 = np.array([255, 255, 160])
#     #
#     # lower_black2 = np.array([40, 0, 0])
#     # upper_black2 = np.array([255, 255, 55])
#     #
#     # lower_black3 = np.array([0, 0, 0])
#     # upper_black3 = np.array([20, 255, 50])
#     floor = cv2.inRange(obs_roi, lower_black1, upper_black1)
#     line = cv2.inRange(obs_roi, lower_yellow1, upper_yellow1)
#     line = line + cv2.inRange(obs_roi, lower_yellow2, upper_yellow2)
#     removeRegion(line, floor)
#     # floor = floor + cv2.inRange(obs_roi, lower_black2, upper_black2)
#     # floor = floor + cv2.inRange(obs_roi, lower_black3, upper_black3)
#
#     kernel = np.ones((5, 5), np.uint8)
#     line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel)
#
#     obstacle = 255 - cv2.bitwise_or(line, floor)
#
#     _, line_contour, _  = find_main_contour(line)
#     cv2.drawContours(image[1 * int(height / 3): height, :], line_contour, -1, (255, 0, 255), 2)
#     obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_CLOSE, kernel)
#     if line_contour is not None:
#         removecontourRegion(obstacle, line_contour)
#     else:
#         obstacle[:,:] = 0
#
#     #obstacle detection
#     _, _, obs_box= find_main_contour_obs(obstacle)
#     if obs_box is not None:
#         dbox = geom.shift_box(obs_box, w_offset, h_offset)
#         cv2.drawContours(image, [dbox], -1, (255, 0, 255), 2)
#
#     #line detection
#     line_height, line_width = line.shape[:2]
#     line_roi = line[1 * int(line_height / 2): line_height, :]
#
#     _, max_cont_line, box = find_main_contour(line_roi)
#
#     if max_cont_line is not None:
#
#         M = cv2.moments(max_cont_line)
#         # print(M['m00'])
#         if M['m00'] > 0.0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#             cv2.circle(image, (cx + w_offset, cy + h_offset), 2, (0, 0, 255), 2)
#             # if box is not None:
#             #     dbox = geom.shift_box(box, w_offset, h_offset)
#             #     # cv2.drawContours(image, [dbox], -1, (255, 0, 255), 2)
#         else:
#             print('No center detected!')
#     else:
#         print('no line found')
#
#     if show:
#         try:
#             cv2.imshow('hsv', obs_roi)
#             cv2.imshow('line', line)
#             cv2.imshow('floor', floor)
#             cv2.imshow('obs', obstacle)
#         except Exception as e:
#             print(e)
#             pass
#
#     return image, cx, cy, obstacleFlag

def points(mask):
    candidates = np.where(mask==255)
    left = candidates[1].min()
    right = candidates[1].max()
    top = candidates[0].min()
    bottom = candidates[0].max()
    return left, right, top, bottom

def removeRegion(mask, refmask):
    refLeft, refRight, refTop, refBottom = points(refmask)
    mask[0:refTop, :] =0
    mask[refBottom:, :] = 0
    mask[:, 0:refLeft] = 0
    mask[:,refRight:] = 0


def removecontourRegion(mask, contour):
    # pdb.set_trace()
    # print(len(contour))
    contourLeft = tuple(contour[contour[:, :, 0].argmin()][0])[1]
    contourRight = tuple(contour[contour[:, :, 0].argmax()][0])[1]
    contourTop = tuple(contour[contour[:, :, 1].argmin()][0])[0]
    contourBottom = tuple(contour[contour[:, :, 1].argmax()][0])[0]
    # print(lineLeft, lineRight, lineTop, lineBot)

    mask[0:contourTop, :] =0
    mask[contourBottom:, :] = 0
    mask[:, 0:contourLeft] = 0
    mask[:,contourRight:] = 0


