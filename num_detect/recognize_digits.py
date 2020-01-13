import numpy as np
import cv2
import pdb
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
org_image = cv2.imread("./exp1.jpg")
org_image = cv2.resize(org_image, (500, 500))
image = cv2.cvtColor(org_image, cv2.COLOR_BGR2HSV)

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (1, 1, 1, 1, 1, 1, 1): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3
}

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

# define the list of boundaries
boundaries = [  #BGR
   # ([17, 15, 100], [50, 56, 200]),  #RED RGB
    # ([0, 100, 100], [10, 255, 255]),  #RED
    ([36, 25, 25], [70, 255,255])  #GREEN
]

# loop over the boundaries
for idx, (lower, upper) in enumerate(boundaries):

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    ratio = image.shape[0] / float(output.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    # gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    h, s, v1 = cv2.split(output)

    blurred = cv2.GaussianBlur(v1, (21, 21), 0)
    thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image and initialize the shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if idx == 0:
    #     if cnts != None:
    #         print("stop")
    #         M = cv2.moments(cnts[0])
    #         cX = int((M["m10"] / M["m00"]) * ratio)
    #         cY = int((M["m01"] / M["m00"]) * ratio)
    #         cv2.drawContours(org_image, [cnts[0]], -1, (0, 255, 0), 2)
    #         cv2.putText(org_image, "stop", (cX - 20, cY - 70), cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.5, (200, 100, 255), 2)
    #     continue
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour

        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape, th_digits = detect(c, v1)
        cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
        cv2.putText(output, shape, (cX - 25, cY - 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        cv2.imshow("images", output)

        digits = cv2.findContours(th_digits.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digits = imutils.grab_contours(digits)

        digitCnts = []
        for cc in digits:
            # extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(cc)
            if (w >= 4 and w <= 40) and (h >= 25 and h <= 30):
                digitCnts.append(cc)

        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
        digits = []
        print("%d digits deteted"%len(digitCnts))

        for cc in digitCnts:

            (x, y, w, h) = cv2.boundingRect(cc)
            roi = th_digits[y:y + h, x:x + w]

            # compute the width and height of each of the 7 segments
            # we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
            dHC = int(roiH * 0.05)

            cv2.rectangle(th_digits, (x, y), (x + w, y + h), (200, 100, 255), 2)
            cv2.imshow("Output", cv2.resize(th_digits, (250,250)))
            cv2.waitKey(0)

            # define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)),  # top
                ((0, 0), (dW, h // 2)),  # top-left
                ((w - dW, 0), (w, h // 2)),  # top-right
                ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
                ((0, h // 2), (dW, h)),  # bottom-left
                ((w - dW, h // 2), (w, h)),  # bottom-right
                ((0, h - dH), (w, h))  # bottom
            ]
            on = [0] * len(segments)

            # loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)

                # if the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if total / float(area) >= 0.6:
                    on[i] = 1

            # lookup the digit and draw it on the image
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append(digit)

            cv2.rectangle(th_digits, (x, y), (x + w, y + h), (200, 100, 255), 2)
            cv2.putText(th_digits, str(digit), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        print("digits: {}{}{}".format(*digits[:3]))
        cv2.imshow("Output", cv2.resize(th_digits, (250,250)))
        cv2.waitKey(0)



