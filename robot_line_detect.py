import cv2
import numpy as np

# 이 부분에 로봇에서 받아온 이미지로 수정해서 추가
datafolder = '/Users/soua/Desktop/Project/RoadImg'
# imgpath = datafolder + '/Img_675.jpg'
# image = cv2.imread(imgpath)
#
# cv2.imshow('road', image)
# # # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # Convert to HSV color space
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv', hsv)
# # Define range of white color in HSV
# # lower_white = np.array([0, 0, 212])
# # upper_white = np.array([131, 255, 255])
# lower_yellow = np.array([20, 40, 240])
# upper_yellow = np.array([230, 255, 255])
# # Threshold the HSV image
# mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# # Remove noise
# kernel_erode = np.ones((4,4), np.uint8)
# eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
# kernel_dilate = np.ones((6,6),np.uint8)
# dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
#
# # Find the different contours
# contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # Sort by area (keep only the biggest one)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
# if len(contours) > 0:
#     M = cv2.moments(contours[0])
#     # Centroid
#     cx = int(M['m10']/M['m00'])
#     cy = int(M['m01']/M['m00'])
#     print("Centroid of the biggest area: ({}, {})".format(cx, cy))
# else:
#     print("No Centroid Found")
#
# cv2.drawContours(image, contours, -1, (255, 0, 0), 3)
# cv2.imshow('Contour', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###############################

imagelist = []
for i in range(604, 1722):
    imgpath = datafolder + '/Img_%d.jpg'%i
    image = cv2.imread(imgpath)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 40, 240])
    upper_yellow = np.array([230, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel_erode = np.ones((4, 4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((6, 6), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    if len(contours) > 0:
        M = cv2.moments(contours[0])

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print("Centroid of the biggest area: ({}, {})".format(cx, cy))
    else:
        print("No Centroid Found")

    cv2.drawContours(image, contours, -1, (255, 0, 0), 3)
    imagelist.append(image)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('../demo20_contour.mp4', fourcc, 20.0, (480,360), True)

for i in range(len(imagelist)):
    # filename ='../Img_%d.jpg'%i
    # img = cv2.imread(filename)
    img = imagelist[i]
    out.write(img)

out.release()
cv2.destroyAllWindows()