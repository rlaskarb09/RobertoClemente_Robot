from imutils import paths
import numpy as np
import imutils
import cv2
import line_detect.geom_util as geom
import matplotlib.pyplot as plt
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage import exposure
import time

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
   avg_color_per_row = np.average(image, axis=0)
   avg_color = np.average(avg_color_per_row, axis=0)
   avg = np.sum(avg_color)/3
   return avg


#################
boundaries = [([30, 150, 100], [70, 250, 200])]   #GREEN

datafolder = '/Users/soua/Desktop/Project/sterling_demo'

imglist = []
timelist = []
for i in range(2657):
    imgPath = datafolder + '/Img_%d.jpg'%i
    img_ = cv2.imread(imgPath)

    start_time = time.time()

    img = edge_enhancement(img_)
    img = cv2.resize(img, (500,500))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # if np.mean(v) < 120:
    #     v = 255 - v
    # else :
    #     v = v + 100
    # if np.mean(s) < 55 :
    #     s = s + 100
    # else:
    #     s = s + 50
    v = 255 - v
    s = s + 100
    hsv = cv2.merge((h, s, v))

    lowerG = np.array(boundaries[0][0], dtype='uint8')
    upperG = np.array(boundaries[0][1], dtype='uint8')
    mask = cv2.inRange(hsv, lowerG, upperG)
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    try:
        c = max(cnts, key=cv2.contourArea)
        mark = cv2.minAreaRect(c)
        box = cv2.boxPoints(mark)
        box = np.int0(box)
        cv2.drawContours(img_, [box], -1, (0, 0, 255), 2)
    except:
        print('There is no signs....')
    timelist.append(time.time()-start_time)
    print('%d th image processing....'%i)
    imglist.append(img_)

print('Avg time for processing : ', (sum(timelist)/len(timelist)))

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('dist_demo.mp4', fourcc, 20.0, (480, 360), True)

for i in range(len(imglist)):
    out.write(imglist[i])

out.release()
cv2.destroyAllWindows()

