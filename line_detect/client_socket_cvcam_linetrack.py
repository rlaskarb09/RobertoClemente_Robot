import threading
import cv2
import time
from socket import *
import queue
from linedetect import *
import numpy as np
import pdb
from movement import *

# # thread Lock:

# threadLock.acquire()
# threadLock.release()

FRAME_RATE = 20
FRAME_TIME = 1 / FRAME_RATE
INIT_TIME_DELAY = 12 * FRAME_TIME

# 7cm

class ReadImageThread(threading.Thread):
    def __init__(self, threadLock, frameQueue, width=320, height=240, frameRate=FRAME_RATE, bufferSize=1):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(0)
        self.encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        self.frameQueue = frameQueue
        # Set the width, height, frame rate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, frameRate)
        # Set the buffer size to 1
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, bufferSize)

    def run(self):
        print('capture start')
        start = time.time()
        while True:
            ret, frame = self.cap.read()
            # ret, encodedFrame = cv2.imencode('.jpg', frame, self.encodeParam)
            self.frameQueue.put(frame)
            # self.frameQueue.put(frame)
        end = time.time()
        print('run time:', end - start)

class CommunicateThread(threading.Thread):
    def __init__(self, threadLock, frameQueue, serverName='172.26.226.69', timeDelay=0.6, showFrame = True, serverPort=8888):
        threading.Thread.__init__(self)
        self.threadLock = threadLock
        self.encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        # self.kit = ServoKit(channels=16)
        self.serverName = serverName
        self.serverPort = serverPort
        self.frameQueue = frameQueue

        #timing
        self.TIME_DELAY = timeDelay
        self.FRAME_TIME = FRAME_TIME

        #sending option
        self.showFrame = showFrame

    def run(self):
        # clientSocket
        self.clientSocket = socket(AF_INET, SOCK_STREAM)
        self.clientSocket.connect((self.serverName, self.serverPort))

        # # Flush Queue safely
        # with self.frameQueue.mutex:
        #     self.frameQueue.clear()
        self.linetrack()

        # Encoding jpg to lower resolution
        # Save to Queue (with threadLock)

    def getFrame(self):
        return self.frameQueue.get()

    def sendFrame(self, frame):
        # Send frame
        ret, encodedFrame = cv2.imencode('.jpg', frame, self.encodeParam)
        encoded = encodedFrame.tostring()
        self.clientSocket.send(str(len(encoded)).encode().ljust(16))
        self.clientSocket.send(encoded)

    def linetrack(self):

        last_shift = 0
        last_angle = 90
        move("STOP")
        while True:
            if not self.frameQueue.empty():
                # first frame
                frame = self.getFrame()
                # sync
                move("FORWARD")
                time.sleep(self.TIME_DELAY)
                last_shift, last_angle = self.getAction(frame, last_shift, last_angle, self.showFrame)
                break
            else:
                 continue
        i=1

        while True:
            if not self.frameQueue.empty():
                if self.frameQueue.qsize() > 1:
                    print('qsize:', self.frameQueue.qsize())
                frame = self.getFrame()
            print('#', i )
            move("FORWARD")
            last_shift, last_angle = self.getAction(frame, last_shift, last_angle, self.showFrame)
            i+=1

    def find_line(self, side):
        # logging.debug(("Finding line", side))
        print("Finding Line")
        if side == 0:
            return None, None

        for i in range(0, conf.find_turn_attempts):
            turn(side, conf.find_turn_step)
            frame = self.getFrame()
            frame, angle, shift = lineDetect(frame)
            self.sendFrame(frame)
            if angle is not None:
                return angle, shift
        return None, None

    def getAction(self, frame, last_shift, last_angle, show):
        startTime = time.time()

        leftSpeed = 0.5
        rightSpeed = 0.5
        rightCenter = conf.rightCenter
        rightAngle = conf.rightAngle

        frame, shift, angle = lineDetect(frame)
        # if angle is None:
        #     move("STOP")

        if show == 'TRUE':
            # send frame
            self.sendFrame(frame)

        if angle is not None:
            err = conf.shift_step * (shift - rightCenter) / 100 + conf.angle_step * (- (angle - rightAngle) / 90)
            der = (shift - last_shift) / 100 -(angle - last_angle) / 90
            PIDf = (err * conf.kp + der * conf.kd) / 2
            print("PIDf:%.5f" % PIDf)
            print("process time:", time.time()-startTime)
            # pdb.set_trace()
            if PIDf > 0:
                print("RIGHT")
                motorSpeed(leftSpeed, rightSpeed - PIDf) #turn right
            else:
                print("LEFT")
                motorSpeed(leftSpeed + PIDf, - rightSpeed)  #turn left
            last_shift = shift
            last_angle = angle

        return last_shift, last_angle


if __name__=='__main__':
    frameQueue = queue.Queue()
    threadLock = threading.Lock()

    thread1 = ReadImageThread(threadLock, frameQueue)
    thread2 = CommunicateThread(threadLock, frameQueue, timeDelay= INIT_TIME_DELAY, showFrame = 'TRUE' , serverName='172.26.225.55')

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

