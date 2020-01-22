import cv2
import num_detect.number_geom as geom

# 이 부분에 로봇에서 받아온 이미지로 수정해서 추가
datafolder = '/Users/soua/Desktop/Project/RoadImg'
imgpath = datafolder + '/Img_675.jpg'
image = cv2.imread(imgpath)
# cv2.imshow('before', image)
w, h = image.shape[:2]
image = geom.edge_enhancement(image)
# cv2.imshow('after', image)

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv', hsv)
hsv = cv2.GaussianBlur(hsv, (9, 9), 0)
edge = cv2.Canny(hsv, 35, 125)
# cv2.imshow('edge', edge)
# cv2.waitKey(0)

cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
cv2.drawContours(image, cnts[:1][0], -1, (255, 0, 0), 2)
cv2.imshow('contours', image)
cv2.waitKey(0)