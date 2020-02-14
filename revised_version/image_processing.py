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
import pdb

ADDRESS_ANTICLOCKWISE = ['101', '102', '103', '203', '202', '201', 'stop']
ADDRESS_CLOCKWISE = ['201', '202', '203', '103', '102', '101', 'stop']

FRAME_INTERVAL_ANTICLOCKWISE = {'101': 32, '102': 56, '103': 56, '203': 66, '202': 56, '201': 56, 'stop': 45}
FRAME_INTERVAL_CLOCKWISE = {'201': 35, '202': 60, '203': 70, '103': 70, '102': 50, '101': 56, 'stop': 35}


class ImageProcessingProcess(Process):
    def __init__(self, frameQueue, sendQueue, status=Manager().dict(), isServerConnect = True):
        super(ImageProcessingProcess, self).__init__()
        self.isServerConnect = isServerConnect
        self.encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        self.frameQueue = frameQueue
        self.sendQueue = sendQueue
        self.status = status
        self.frameWidth = 0
        self.frameHeight = 0

        #stop option
        self.stopCount = 0
        self.willStop = False

        #obstacle option
        self.obstacleStopCount = 0
        self.willobstacleStop = False

        #address option
        self.addressStopCount = 0
        self.willaddressStop = False

        # address setting
        self.address = ADDRESS_ANTICLOCKWISE
        self.addressInterval = FRAME_INTERVAL_ANTICLOCKWISE
        self.directionFlag = 1  # 1 (anticlockwise)  /  -1 (clockwise)
        self.last_cx = 0

    def run(self):
        move(0, 0)
        self.robotRun()

    def storeFrame(self, frame):
        ret, encodedFrame = cv2.imencode('.jpg', frame, self.encodeParam)
        self.sendQueue.put(encodedFrame)

    def getFrame(self):
        return self.frameQueue.get()

    def robotRun(self):
        # Annealing for the first time
        startTime = time.time()
        while time.time() - startTime < 2.0:
            if not self.frameQueue.empty():
                frame = self.frameQueue.get()
                self.frameWidth, self.frameHeight = frame.shape[:2]
                self.detection(self.frameQueue.get())

        self.stopSTOP()
        last_cx = 0

        # timing option
        self.timingIndex = 0
        self.frame_num = 0
        self.curr_address = self.address[self.timingIndex]
        self.frame_to_go = self.addressInterval[self.curr_address]

        while True:
            if not self.frameQueue.empty():
                frame = self.getFrame()
                if self.status['command'] == 'maintenance':
                    self.stopMaintenance()
                else:
                    self.status['mode'] = 'move'
                    processingStart = time.time()
                    newFrame, cx, cy, obstacleFlag, stopFlag, addressFlag = self.detection(frame)

                    self.stopProcedure(stopFlag)
                    self.obstacleProcedure(obstacleFlag, stopFlag)
                    # self.addressProcedure(addressFlag) or
                    if self.frame_num == self.frame_to_go:
                        self.frame_num = 0
                        if self.timingIndex < len(self.address):
                            curr_address = self.address[self.timingIndex]
                            print('current address:', curr_address)
                            self.status['location'] = curr_address
                            self.timingIndex = (self.timingIndex + 1)
                            if self.timingIndex < len(self.address):
                                next_address = self.address[self.timingIndex]
                                self.frame_to_go = self.addressInterval[next_address]
                            else:
                                self.frame_to_go = 100000
                        if curr_address in self.status['path']:
                            self.stopAddress(curr_address)

                    if cx is not None:
                        self.lineTrace(cx)
                        self.frame_num += 1
                        self.last_cx = cx
                    # else:
                    #     print('no line found!')
                    #     move(0.0, 0.0)
                    #     line_found = self.findLine(self.frameWidth)
                    #     if not line_found:
                    #         print('no line found!')
                    #         self.stopMaintenance()

                    processingEnd = time.time()

                    self.storeFrame(frame)
                    # print('processing time:', processingEnd - processingStart)

    def detection(self, frame):
        addressFlag =False
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_line, cx, cy, obstacleFlag= lineDetectObstacle(frame, hsv)
        frame_new, stopFlag = stopDetect(frame_line, hsv)
        # lab = prepare_addr_pic(frame)
        # frame_new, addressFlag = greenaddressDetect(frame, lab)
        return frame_new, cx, cy, obstacleFlag, stopFlag, addressFlag

    def lineTrace(self, cx):
        shiftX = self.getShiftX(cx)
        self.moveMotor(shiftX)

    def getShiftX(self, cx):
        # centerX = 3* self.frameWidth / 4
        # centerX = 1* self.frameWidth /2
        if self.directionFlag == 1 :
            centerX = 5* self.frameWidth /8
        else:
            centerX = 3* self.frameWidth /8


        return cx - centerX

    def moveMotor(self, shiftX):
        maxSpeed = 0.40
        minSpeed = 0.0
        halfWidth = self.frameWidth / 2
        if shiftX > 0:
            move(maxSpeed, minSpeed + ((maxSpeed - minSpeed) * (halfWidth - shiftX) / halfWidth) + 0.05)
        elif shiftX < 0:
            move(minSpeed + ((maxSpeed - minSpeed) * (halfWidth + shiftX) / halfWidth), maxSpeed + 0.05)
        else:
            move(maxSpeed, maxSpeed + 0.05)

    def stopProcedure(self, stopFlag, stopCountNum=9, willStopCountNum = 20):
        if stopFlag and not self.willStop:  # stop detected
            self.stopCount += 1
        if self.stopCount >= stopCountNum:
            self.stopCount = 0
            self.willStop = True
            self.willStopCount = willStopCountNum
        if self.willStop:
            print('self.willStopCount:', self.willStopCount)
            self.willStopCount -= 1
        if (self.willStop and self.willStopCount <= 0) or (self.address[self.timingIndex] == 'stop' and self.frame_num == self.frame_to_go):
            self.stopSTOP()
            self.willStop = False

    def obstacleProcedure(self, obstacleFlag, stopFlag, obstacleCountNum=1, willStopCountNum = 0):
        # and not stopFlag
        if obstacleFlag  and not self.willobstacleStop:  # stop detected
            self.obstacleStopCount += 1
        if self.obstacleStopCount >= obstacleCountNum:
            self.obstacleStopCount = 0
            self.willobstacleStop = True
            self.willObstacleCount = willStopCountNum
        if self.willobstacleStop:
            print('self.willobstacleCount:', self.willObstacleCount)
            self.willObstacleCount -= 1
        if self.willobstacleStop and self.willObstacleCount <= 0:
            self.stopMaintenance()
            # self.stopObstacle()
            self.willobstacleStop = False

    def addressProcedure(self, addressFlag, addressCountNum=1, willStopCountNum =11):
        if addressFlag and not self.willaddressStop:  # address detected
            self.addressStopCount += 1
        if self.addressStopCount >= addressCountNum:
            self.addressStopCount = 0
            self.willaddressStop = True
            self.willAddressCount = willStopCountNum
        if self.willaddressStop:
            print('self.addressCount:', self.willAddressCount)
            self.willAddressCount -= 1
        if self.willaddressStop and self.willAddressCount <= 0:
            self.willaddressStop = False
            return True
        else:
            return False

    def stopSTOP(self):
        start_time = time.time()
        move(0, 0)
        print('stop at STOP')
        print('current address: stop')
        self.status['mode'] = 'stop'
        self.status['location'] = 'stop'
        if self.isServerConnect:
            while True:
                if self.status['command'] == 'move':
                    self.timingIndex = 0
                    self.frame_num = 0
                    self.curr_address = self.address[self.timingIndex]
                    self.frame_to_go = self.addressInterval[self.curr_address]
                    if self.status['path'] == ['101'] :
                        if self.directionFlag == -1:
                            self.turn(1.2)
                    elif self.status['path'] == ['201'] :
                        if self.directionFlag == 1:
                            self.turn(1.2)
                    else:
                        if self.directionFlag == -1:
                            self.turn(1.2)
                    break
                elif self.status['command'] == 'maintenance':
                    self.stopMaintenance()
                    break
                else:
                    if not self.frameQueue.empty():
                        self.getFrame()
        else:
            #or self.status['path'] == ['101', '102']
            #or self.status['path'] == ['201', '202']
            while True:
                if time.time() - start_time < 2:
                    if not self.frameQueue.empty():
                        self.getFrame()
                elif self.status['command'] == 'maintenance':
                    self.stopMaintenance()
                    break
                else:
                    print('current address: stop')
                    self.status['mode'] = 'stop'
                    self.status['location'] = 'stop'
                    self.timingIndex = 0
                    self.frame_num = 0
                    self.curr_address = self.address[self.timingIndex]
                    self.frame_to_go = self.addressInterval[self.curr_address]
                    if self.status['path'] == ['101'] :
                        if self.directionFlag == -1:
                            self.turn(1.2)
                    elif self.status['path'] == ['201'] :
                        if self.directionFlag == 1:
                            self.turn(1.2)
                    else:
                        if self.directionFlag == -1:
                            self.turn(1.2)
                    break

    def stopAddress(self, address):
        start_time = time.time()
        move(0, 0)
        self.status['mode'] = 'stop'
        print('stop at address', address)

        if self.isServerConnect:
            while True:
                if self.status['command'] == 'move':
                    if self.status['path'] == ['101']:
                        self.turn(1.3)
                    elif self.status['path'] == ['201']:
                        self.turn(1.4)
                    # elif self.status['path'] == ['101', '102'] and address == '102':
                    #     self.turn(1.3)
                    # elif self.status['path'] == ['201', '202'] and address == '202':
                    #     self.turn(1.3)
                    break
                elif self.status['command'] == 'maintenance':
                    self.stopMaintenance()
                    break
                else:
                    if not self.frameQueue.empty():
                        self.getFrame()
                    continue
        else:
            while True:
                if time.time() - start_time < 2:
                    if not self.frameQueue.empty():
                        self.getFrame()
                elif self.status['command'] == 'maintenance':
                    self.stopMaintenance()
                    break
                else:
                    if self.status['path'] == ['101']:
                        self.turn(1.3)
                    elif self.status['path'] == ['201']:
                        self.turn(1.4)
                    # elif self.status['path'] == ['101', '102'] and address == '102':
                    #     self.turn(1.3)
                    # # elif self.status['path'] == ['201', '202'] and address == '202':
                    #     self.turn(1.3)
                    break

    def stopMaintenance(self):
        move(0, 0)
        self.status['mode'] = 'maintenance'
        print('maintenance mode')
        while True:
            if self.status['command'] == 'move':
                break
            else:
                if not self.frameQueue.empty():
                    self.getFrame()
                continue

    def stopObstacle(self):
        move(0, 0)
        self.status['mode'] = 'maintenance'
        # print('obstacle found')
        while True:
            if not self.frameQueue.empty():
                frame = self.getFrame()
                newFrame, cx, cy, obstacleFlag, stopFlag, addressFlag = self.detection(frame)
                self.storeFrame(newFrame)
                if cx is not None or stopFlag or self.status['command'] == 'maintenance':
                    break

    def findLine(self, last_cx):
        lineFound = False
        print("Finding Line")
        if last_cx <= self.frameWidth:
            side = 'left'
        else:
            side = 'right'
        print(last_cx, side)
        delay = 2.4/20

        for i in range(0, 20):
            start = time.time()
            turn(side)
            while time.time() - start < delay:
                if not self.frameQueue.empty():
                    self.getFrame()
            move(0.0, 0.0)
            while True:
                if not self.frameQueue.empty():
                    frame = self.getFrame()
                    newFrame, cx, cy, obstacleFlag, stopFlag, addressFlag = self.detection(frame)
                    self.storeFrame(newFrame)
                    if cx is not None:
                        lineFound = True
                        print('line Founded at %f degrees' % float((360/20)*(i+1)))
                        self.lineTrace(cx)
                        return lineFound
                    break
                else:
                    continue
        return lineFound

    def turn(self, delay):
        start = time.time()
        self.directionFlag = self.directionFlag * -1
        print(self.directionFlag)
        if self.directionFlag == 1:
            self.address = ADDRESS_ANTICLOCKWISE
            self.addressInterval = FRAME_INTERVAL_ANTICLOCKWISE
            side = 'left'
        else:
            self.address = ADDRESS_CLOCKWISE
            self.addressInterval = FRAME_INTERVAL_CLOCKWISE
            side = 'right'

        while time.time() - start < delay:
            if not self.frameQueue.empty():
                self.getFrame()
            turn(side)
        move(0.0, 0.0)

