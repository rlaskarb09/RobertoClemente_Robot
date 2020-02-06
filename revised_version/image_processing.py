from multiprocessing import Process, Manager
import cv2
from line_detect.linedetect import *
import line_detect.conf as conf
from collections import deque
# from line_detect.movement import *
from num_detect.stopdetect import *
from num_detect.detect_stop_addr_function import *
import time
from movement import *

class ImageProcessingProcess(Process):
    def __init__(self, frameQueue, sendQueue, status=Manager().dict()):
        super(ImageProcessingProcess, self).__init__()
        self.encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        self.frameQueue = frameQueue
        self.sendQueue = sendQueue
        self.status = status
        self.frameWidth = 0
        self.frameHeight = 0

    def run(self):
        move(0, 0)
        self.linetrack()

    def storeFrame(self, frame):
        ret, encodedFrame = cv2.imencode('.jpg', frame, self.encodeParam)
        self.sendQueue.put(encodedFrame)

    def getFrame(self):
        return self.frameQueue.get()

    def linetrack(self):
        # Annealing for the first time
        startTime = time.time()
        while time.time() - startTime < 1.0:
            if not self.frameQueue.empty():
                frame = self.frameQueue.get()
                self.frameWidth, self.frameHeight = frame.shape[:2]
                self.detectLine(self.frameQueue.get())

        while True:
            if not self.frameQueue.empty():
                frame = self.getFrame()
                processingStart = time.time()
                cx, cy, newFrame = self.detectLine(frame)
                shiftX = self.getShiftX(cx)
                self.moveMotor(shiftX)
                processingEnd = time.time()
                self.storeFrame(newFrame)
                print('processing time:', processingEnd - processingStart)
            
    def detectLine(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        crop = hsv[2 * int(height / 3): height, :]
        lower_yellow = np.array([20, 30, 150])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(crop, lower_yellow, upper_yellow)
        cnts, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if cnts is not None and len(cnts) > 0:
            C = max(cnts, key=cv2.contourArea)
            M = cv2.moments(C)
            if M['m00'] > 0.0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            else:
                crop_height, crop_width = crop.shape[:2]
                cx, cy = int(crop_width / 2), int(crop_height / 2)
        else:
            crop_height, crop_width = crop.shape[:2]
            cx, cy = int(crop_width / 2), int(crop_height / 2)

        try:
            crop = cv2.circle(crop, (cx, cy), 2, (255, 0, 0), 2)
            print('cx, cy =', cx, cy)
        except:
            print('exception in cv2.circle. cx, cy =', cx, cy)

        
        return cx, cy, crop

    def getShiftX(self, cx):
        centerX = self.frameWidth / 2
        return cx - centerX

    def moveMotor(self, shiftX):
        maxSpeed = 0.30
        minSpeed = 0.00
        halfWidth = self.frameWidth / 2

        if shiftX > 0:
            move(maxSpeed, minSpeed + ((maxSpeed - minSpeed) * (halfWidth - shiftX) / halfWidth) + 0.05)
        elif shiftX < 0:
            move(minSpeed + ((maxSpeed - minSpeed) * (halfWidth + shiftX) / halfWidth), maxSpeed + 0.05)            
        else:
            move(maxSpeed, maxSpeed + 0.05)
        