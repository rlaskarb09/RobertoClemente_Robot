import logging
import os
import threading
import cv2
import time
from websocket_multi_thread import WebSocketThread
from socket import *
from collections import deque
import queue
from line_detect.linedetect import *
import pdb
import line_detect.conf as conf
from line_detect.movement import *
from num_detect.detect_stop_addr_function import *

# # thread Lock:

# threadLock.acquire()
# threadLock.release()

FRAME_RATE = 20
FRAME_TIME = 1 / FRAME_RATE
INIT_TIME_DELAY = 12 * FRAME_TIME # 7cm 0.6s
INIT_TIME_DELAY = 0.6 # 7cm 0.6s


ADDRESS_CLOCKWISE = ['201', '202', '203', '103', '102', '101']
ADDRESS_ANTICLOCKWISE = ['101', '102', '103', '203', '202', '201']
STOP_QUEUE_SIZE = 5
ADDRESS_QUEUE_SIZE = 5
OBSTACLE_QUEUE_SIZE = 5

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
        logging.info('capture start')
        start = time.time()
        while True:
            ret, frame = self.cap.read()
            self.frameQueue.put(frame)
        end = time.time()
        logging.info('capture run time:', end - start)

class CommunicateThread(threading.Thread):
    def __init__(self, threadLock, frameQueue, serverName='172.26.226.69', timeDelay=0.6, showFrame = True, serverPort=8888, status=None):
        threading.Thread.__init__(self)
        self.threadLock = threadLock
        self.encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        # self.kit = ServoKit(channels=16)
        self.serverName = serverName
        self.serverPort = serverPort
        self.status = status
        self.frame = None
        self.frameQueue = frameQueue

        # address setting
        self.address = ADDRESS_ANTICLOCKWISE
        self.directionFlag = -1

        #detection setting
        self.stopQueue = deque(maxlen = STOP_QUEUE_SIZE)
        self.addressQueue = deque(maxlen = ADDRESS_QUEUE_SIZE)
        self.obstacleQueue = deque(maxlen = OBSTACLE_QUEUE_SIZE)

        #timing
        self.TIME_DELAY = timeDelay
        self.FRAME_TIME = FRAME_TIME

        #sending option
        self.showFrame = showFrame
        self.frameNum = 0

        #receiving option
        self.actionLength = 11

    def run(self):
        # clientSocket
        self.clientSocket = socket(AF_INET, SOCK_STREAM)
        self.clientSocket.connect((self.serverName, self.serverPort))
        try:
            self.robotRun()
        except ConnectionRefusedError:  #FIXME
            logging.error('server connection refused!')
            self.stopMaintanence(self.frame)
        # except OSError:     #FIXME
        finally:
            move("STOP")

    def getFrame(self):
        return self.frameQueue.get()

    def sendFrame(self, frame):
        # Send frame
        ret, encodedFrame = cv2.imencode('.jpg', frame, self.encodeParam)
        encoded = encodedFrame.tostring()
        self.clientSocket.send(str(len(encoded)).encode().ljust(16))
        self.clientSocket.send(encoded)
        logging.info('frame: %d' % self.frameNum)
        self.frameNum += 1

    def flushFrame(self):
        with self.frameQueue.mutex:
            self.frameQueue.queue.clear()

    # def synchronization(self, frame, last_shift, last_angle):
    #     self.status['mode'] = 'move'
    #     self.status['path'].append('201')
    #     # sync
    #     move("FORWARD")
    #     time.sleep(self.TIME_DELAY)
    #     last_shift, last_angle = self.getAction(frame, last_shift, last_angle, self.showFrame)
    #     self.flushFrame()

    def robotRun(self):
        last_shift = 0
        last_angle = 90
        frame_num = 0

        move("STOP")
        self.status['mode'] ='stop'
        self.status['location'] ='stop'
        while True:
            if not self.frameQueue.empty():
                # first frame
                self.frame = self.getFrame()

                if self.status['command'] == 'move':
                    self.status['mode'] = 'move'
                    # sync  # FIXME
                    curr_shift, curr_angle = self.detection(self.frame)
                    move("FORWARD")
                    time.sleep(self.TIME_DELAY)         # FIXME detection time 보고 delay 조정
                    # curr_shift, curr_angle = self.detection(self.frame)
                    last_shift, last_angle = self.lineTrace(curr_shift, curr_angle, last_shift, last_angle)
                    self.flushFrame()
                    frame_num += 1
                    move("STOP")
                    break

            else:
                 continue

        while True:
            keyMessage = self.clientSocket.recv(self.actionLength)
            key = keyMessage.decode().strip()
            if key == 'KEY_PRESSED':
                move("STOP")
                logging.debug('message from the server: stop')

                while True:
                    self.sendFrame(self.frame)
                    keyMessage = self.clientSocket.recv(self.actionLength)
                    key = keyMessage.decode().strip()
                    if key == 'NO_KEY':
                        logging.debug('message from the server: move again')
                        break
                    else:
                        continue

            if not self.frameQueue.empty():
                self.frame = self.getFrame()
                if self.frameQueue.qsize() >= 1:        #FIXME
                    logging.info('qsize: %d'% self.frameQueue.qsize())
                    self.flushFrame()
                frame_num += 1

                if self.status['command'] == 'maintenance':
                    self.stopMaintenance(self.frame)
                else:
                    self.status['mode'] = 'move'
                    move("FORWARD")
                    # detection
                    curr_shift, curr_angle = self.detection(self.frame)
                    # action
                    if self.stopQueue.count(True) > len(self.stopQueue) / 2:        # stop detected
                        self.stopQueue = []
                        self.stopSTOP(self.frame)
                    elif self.addressQueue.count(True) > len(self.addressQueue) / 2:        # address detected
                        self.addressQueue = []
                        address = self.address.pop(0)
                        self.status['location'] = address
                        if address in self.status['path']:
                            self.stopAddress(self.frame, address)
                        else:
                            last_shift, last_angle = self.lineTrace(curr_shift, curr_angle, last_shift, last_angle)
                    else:
                        last_shift, last_angle = self.lineTrace(curr_shift, curr_angle, last_shift, last_angle)

    def find_line(self, side):
        logging.debug("Finding Line")
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

    def detection(self, frame):
        stopFlag, addressFlag = None, None
        startTime = time.time()
        hsv = prepare_stop_pic(frame)
        frame_line, shift, angle, obstacleFlag = lineDetect(frame, hsv)
        lineTime = time.time()
        # frame_stop, stopFlag = stopDetect(frame_line, hsv)
        stopTime = time.time()
        # lab = prepare_addr_pic(frame)
        # frame_add, addressFlag = greenaddressDetect(frame_line, lab)
        addressTime = time.time()
        self.sendFrame(frame_line)  # send frame
        logging.debug('detection | angle: %s, shift: %s, obstacle: %s, stop:%s, address:%s'
                      % (angle, shift, obstacleFlag, stopFlag, addressFlag))
        logging.debug('detection time | line+obstacle: %s sec, stop: %s sec, address: %s sec, total: %s sec'
                      % (lineTime - startTime, stopTime - lineTime, addressTime - stopTime, time.time() - startTime))
        self.obstacleQueue.append(obstacleFlag)
        self.stopQueue.append(stopFlag)
        self.addressQueue.append(addressFlag)
        return shift, angle

    def lineTrace(self, curr_shift, curr_angle, last_shift, last_angle):

        leftSpeed = 0.5
        rightSpeed = 0.5
        rightCenter = conf.rightCenter
        rightAngle = conf.rightAngle

        if curr_angle is not None:  #line found
            err = conf.shift_step * (curr_shift - rightCenter) / 100 + conf.angle_step * (- (curr_angle - rightAngle) / 90)
            der = conf.shift_step *(curr_shift - last_shift) / 100 + conf.angle_step * (-(curr_angle - last_angle)/ 90)
            PIDf = (err * conf.kp + der/4 * conf.kd) / 2
            logging.debug("linetrace | err:  %.5f, der = %.5f, PIDf:%.5f" % (err, der, PIDf))
            if PIDf > 0:
                # print("RIGHT")
                motorSpeed(leftSpeed, rightSpeed - PIDf) #turn right
            else:
                # print("LEFT")
                motorSpeed(leftSpeed + 1.3 * PIDf, - rightSpeed)  #turn left
                # motorSpeed(-leftSpeed + PIDf, - rightSpeed)  #turn left

            last_shift = curr_shift
            last_angle = curr_angle
        else:   #line not found
            logging.debug("linetrace | no line found")
            # move("BACKWARD")    #FIXME
            #obstacle detection
        return last_shift, last_angle

    def stopMaintenance(self, curr_frame):
        move("STOP")
        self.status['mode'] = 'maintenance'
        print('maintenance mode')
        while True:
            if self.status['command'] == 'move':
                self.flushFrame()
                self.frameQueue.put(curr_frame)
                break
            else:
                continue

    def stopSTOP(self, curr_frame):
        move("STOP")
        print('current address: stop')
        self.status['mode'] = 'stop'
        self.status['location'] = 'stop'
        self.status['path'] = []

        # FIXME
        if self.status['path'][0][0] == 1:
            curr_flag = -1
            self.address = ADDRESS_ANTICLOCKWISE
        else:
            curr_flag = 1
            self.address = ADDRESS_CLOCKWISE
        if curr_flag != self.directionFlag:
            self.directionFlag = curr_flag
            turn180()

        while True:
            if self.status['command'] == 'move':
                self.flushFrame()
                self.frameQueue.put(curr_frame)
                break
            else:
                continue

    def stopAddress(self, curr_frame, address):
        move("STOP")
        self.status['mode'] = 'stop'
        # self.status['path'].remove(address)
        print('current address:', address)
        while True:
            if self.status['command'] == 'move':
                self.flushFrame()
                self.frameQueue.put(curr_frame)
                break
            else:
                continue

if __name__=='__main__':
    #log option
    log_folder = './log'
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    timestr = time.strftime("/%Y%m%d_%H%M%S.log")
    logging.basicConfig(filename=log_folder + timestr, level=logging.DEBUG, format='[%(asctime)s][%(levelname)s|%(threadName)s] >> %(message)s')

    frameQueue = queue.Queue()
    threadLock = threading.Lock()
    status = {'command': 'move', 'mode': 'stop', 'location':'stop', 'path':[]}  #FIXME

    thread1 = ReadImageThread(threadLock, frameQueue)
    thread2 = CommunicateThread(threadLock, frameQueue, timeDelay= INIT_TIME_DELAY, showFrame =True, serverName='172.26.225.55', status=status)
    # thread3 = WebSocketThread(status=status)

    try:
        thread1.start()
        thread2.start()
        # thread3.start()
    except KeyboardInterrupt:
        move("STOP")
        thread1.join()
        thread2.join()
        # thread3.join()

