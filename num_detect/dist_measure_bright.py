import numpy as np
import cv2
import num_detect.number_geom as geom
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity


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

        cnts, max_cont, box = geom.find_main_contour(thresh.copy())
        c = max(cnts, key = cv2.contourArea)
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

def yen_bright(image):
    yen_threshold = threshold_yen(image)
    bright_img = rescale_intensity(image, (0, yen_threshold), (0, 255))
    return bright_img

def brightness(image):
   # stat = ImageStat.Stat(image)
   # r,g,b = stat.mean
   # return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
   avg_color_per_row = np.average(image, axis=0)
   avg_color = np.average(avg_color_per_row, axis=0)
   avg = np.sum(avg_color)/3
   return avg

#################
boundaries = [([30, 150, 100], [70, 250, 200])]  #GREEN

datafolder = '/Users/soua/Desktop/Project/sterling_demo'
img_path = datafolder + '/Img_390.jpg'
img = cv2.imread(img_path)
img = edge_enhancement(img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('before hsv', hsv)
h, s, v = cv2.split(hsv)
print('v average : ', np.mean(v))
v = 255 - v
s = s + 100
hsv = cv2.merge((h, s, v))
cv2.imshow('after hsv', hsv)

lowerG = np.array(boundaries[0][0], dtype = 'uint8')
upperG = np.array(boundaries[0][1], dtype = 'uint8')
mask = cv2.inRange(hsv, lowerG, upperG)
cv2.imshow('mask', mask)

output = cv2.bitwise_and(img, img, mask = mask)
cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key = cv2.contourArea)
mark = cv2.minAreaRect(c)
box = cv2.boxPoints(mark)
box = np.int0(box)
cv2.drawContours(img, [box], -1, (0, 0, 255), 2)
cv2.imshow('contour', img)
cv2.waitKey(0)
