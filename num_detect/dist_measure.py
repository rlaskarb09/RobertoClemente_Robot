from imutils import paths
import numpy as np
import imutils
import cv2
import line_detect.geom_util as geom
import matplotlib.pyplot as plt

def find_marker_(image):
    # convert the image to grayscale, blur it, and detect edges
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5,5), 0)
    edged = cv2.Canny(hsv, 35, 125)
    cv2.imshow('edge', edged)
    cv2.waitKey(0)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, cnts[:1][0], -1, (255, 0, 0), 2)
    cv2.imshow('contours', image)
    cv2.waitKey(0)

    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea) # largest area 가 아닌 경우도 있을 수 있음 일단 이 경우에는 제일 큰 area 로 고려해

    return cv2.minAreaRect(c)

def find_marker(image, boundary):
    for idx, (lower, upper) in enumerate(boundary):
        lower = np.array(lower, dtype = 'uint8')
        upper = np.array(upper, dtype = 'uint8')

        # Threshold the HSV image
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)

        h, s, v = cv2.split(output)

        blurred = cv2.GaussianBlur(v, (3, 3), 0)
        thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

        # cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
        cnts, max_cont, box = geom.find_main_contour(thresh.copy())
        c = max(cnts, key = cv2.contourArea)
        # cnts = imutils.grab_contours(cnts)
        return cv2.minAreaRect(c)

def distance_to_camera(knownW, F, perW):
    # compute and return the distance from the marker to the camera
    return (knownW * F) / perW

def adjust_brightness(img, level):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b = np.mean(img[:,:,2])
    if b == 0:
        return img
    r = level / b
    c = hsv.copy()
    c[:,:,2] = c[:,:,2] * r

    return cv2.cvtColor(c, cv2.COLOR_HSV2BGR)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def edge_enhancement(image):
    kernel = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]])/8.0
    output = cv2.filter2D(image, -1, kernel)
    return output

boundaries = [([60, 130, 30], [110, 255, 150])]  #GREEN

#################
datafolder = '/Users/soua/Desktop/Project/distance'
img30_path = datafolder + '/Img_4.jpg'
img30 = cv2.imread(img30_path)
img30 = adjust_gamma(img30, gamma=1.5)
img30 = edge_enhancement(img30)
# histg = cv2.calcHist([img30], [0], None, [256], [0,256])
# plt.plot(histg)
# plt.show()

img30 = cv2.resize(img30, (500,500))
# cv2.imshow('img30', img30)
# cv2.waitKey(0)
hsv30 = cv2.cvtColor(img30, cv2.COLOR_BGR2HSV)

mark30 = find_marker(hsv30, boundaries)
mark30 = cv2.boxPoints(mark30) if imutils.is_cv2() else cv2.boxPoints(mark30)
mark30 = np.int0(mark30)

known_dist = 30/(2.54)
known_width = 2.4/2.54
# find focal length by using img 30cm
focalLenth = (mark30[1][0] * known_dist) / known_width

# cv2.drawContours(img30, [mark30], -1, (0, 0, 255), 2)
# cv2.putText(img30, '%.2f cm' %(known_dist * 2.54), (img30.shape[1]-200, img30.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),3)
# cv2.imshow('img30 text', img30)
# cv2.waitKey(0)

img15_path = datafolder + '/Img_152.jpg'
img15 = cv2.imread(img15_path)
img15 = adjust_gamma(img15, gamma=2.5)
img15 = edge_enhancement(img15)
histg = cv2.calcHist([img15], [0], None, [256], [0,256])
plt.plot(histg)
plt.show()

img15 = cv2.resize(img15, (500, 500))
cv2.imshow('img15', img15)
cv2.waitKey(0)
hsv15 = cv2.cvtColor(img15, cv2.COLOR_BGR2HSV)

mark15 = find_marker(hsv15, boundaries)
inches = distance_to_camera(known_width, focalLenth, mark15[1][0])
box = cv2.boxPoints(mark15) if imutils.is_cv2() else cv2.boxPoints(mark15)
box = np.int0(box)

cv2.drawContours(img15, [box], -1, (0, 0, 255), 2)
cv2.putText(img15, '%.2f cm' %(inches * 2.54), (img15.shape[1] - 200, img15.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
cv2.imshow('image', img15)
cv2.waitKey(0)