import threading
import cv2
import time
import pickle
import numpy
from socket import *
from adafruit_servokit import ServoKit
import queue
import numpy as np

# # thread Lock:

# threadLock.acquire()
# threadLock.release()

class ReadImageThread(threading.Thread):
    def __init__(self, threadLock, frameQueue, width=320, height=240, frameRate=40, bufferSize=1):
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
        for i in range(2000):
            ret, frame = self.cap.read()
            ret, encodedFrame = cv2.imencode('.jpg', frame, self.encodeParam)
            self.frameQueue.put(encodedFrame)
            # self.frameQueue.put(frame)
        end = time.time()
        print('run time:', end - start)

class CommunicateThread(threading.Thread):
    def __init__(self, threadLock, frameQueue, serverName='172.26.226.69', serverPort=8888):
        threading.Thread.__init__(self)
        self.threadLock = threadLock
        self.encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        self.kit = ServoKit(channels=16)
        self.serverName = serverName
        self.serverPort = serverPort
        self.frameQueue = frameQueue
        # actions
        self.FORWARD = 'FORWARD'
        self.BACK = 'BACK'
        self.LEFT = 'LEFT'
        self.RIGHT = 'RIGHT'
        self.STOP = 'STOP'
        self.NO_ACTION = 'NO_ACTION'
        self.ACTION_LENGTH = 10

    def run(self):
        clientSocket = socket(AF_INET, SOCK_STREAM)
        clientSocket.connect((self.serverName, self.serverPort))

        # # Flush Queue safely
        # with self.frameQueue.mutex:
        #     self.frameQueue.clear()
        while True:
            if not self.frameQueue.empty():
                if self.frameQueue.qsize() > 1:
                    print('qsize:', self.frameQueue.qsize())
                frame = self.frameQueue.get()
            else:
                continue
            # Send frame
            encoded = frame.tostring()
            clientSocket.send(str(len(encoded)).encode().ljust(16))
            clientSocket.send(encoded)

            # actionMessage = clientSocket.recv(self.ACTION_LENGTH)
            # action = actionMessage.decode().strip()
            # if action != self.NO_ACTION:
            #     print('Reply from server:', action)
            
            # if action == self.FORWARD:
            #     moveForward(kit)
            # elif action == self.BACK:
            #     moveBack(kit)
            # elif action == self.LEFT:
            #     moveLeft(kit)
            # elif action == self.RIGHT:
            #     moveRight(kit)
            # elif action == self.STOP:
            #     moveStop(kit)
            

        # Encoding jpg to lower resolution
        # Save to Queue (with threadLock)

    def moveForward(self):
        self.kit.continuous_servo[0].throttle =0.5
        self.kit.continuous_servo[1].throttle = -0.5

    def moveBack(self):
        self.kit.continuous_servo[0].throttle = -0.5
        self.kit.continuous_servo[1].throttle = 0.5

    def moveLeft(self):
        self.kit.continuous_servo[0].throttle = -0.05
        self.kit.continuous_servo[1].throttle = -0.4

    def moveRight(self):
        self.kit.continuous_servo[0].throttle = 0.4
        self.kit.continuous_servo[1].throttle = 0.05

    def moveStop(self):
        self.kit.continuous_servo[0].throttle = 0
        self.kit.continuous_servo[1].throttle = 0


if __name__=='__main__':
    frameQueue = queue.Queue()
    threadLock = threading.Lock()

    thread1 = ReadImageThread(threadLock, frameQueue)
    thread2 = CommunicateThread(threadLock, frameQueue, serverName='172.26.226.69')

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()