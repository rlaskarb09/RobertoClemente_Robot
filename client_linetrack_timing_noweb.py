import threading
import os
import cv2
import time
from websocket_multi_thread import WebSocketThread
from socket import *
import queue
from line_detect.linedetect import *
from num_detect.stopdetect import *
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

ADDRESS_CLOCKWISE = ['201', '202', '203', '103', '102', '101', 'stop']
# ADDRESS_CLOCKWISE = ['201', '202', '203', '103', '102', '101']
ADDRESS_ANTICLOCKWISE = ['101', '102', '103', '203', '202', '201', 'stop']
ADDRESS =  ADDRESS_CLOCKWISE
NUM_ADDRESS = len(ADDRESS)
FRAME_INTERVAL = {
    '201': 30,
    '202': 50,
    '203': 60,
    '103': 65,
    '102': 65,
    '101': 60,
    'stop': 35
}

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
    def __init__(self, threadLock, frameQueue, serverName='172.26.226.89', timeDelay=0.6, showFrame = True, serverPort=8888, status=None):
        threading.Thread.__init__(self)
        self.threadLock = threadLock
        self.encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        # self.kit = ServoKit(channels=16)
        self.serverName = serverName
        self.serverPort = serverPort
        self.frameQueue = frameQueue
        self.status = status

        #timing
        self.TIME_DELAY = timeDelay
        self.FRAME_TIME = FRAME_TIME

        #sending option
        self.showFrame = showFrame
        self.actionLength =11

    def run(self):
        # clientSocket
        self.clientSocket = socket(AF_INET, SOCK_STREAM)
        self.clientSocket.connect((self.serverName, self.serverPort))

        self.linetrack()

    def getFrame(self):
        return self.frameQueue.get()

    def sendFrame(self, frame):
        # Send frame
        ret, encodedFrame = cv2.imencode('.jpg', frame, self.encodeParam)
        encoded = encodedFrame.tostring()
        self.clientSocket.send(str(len(encoded)).encode().ljust(16))
        self.clientSocket.send(encoded)

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

    def linetrack(self):
        start = time.time()
        last_shift = 0
        last_angle = 90
        move("STOP")
        self.status['mode'] ='stop'
        self.status['location'] ='stop'
        while True:
            if not self.frameQueue.empty():
                # first frame
                frame = self.getFrame()
                # if self.status['command'] == 'move':
                #     self.status['mode'] = 'move'
                #     self.status['path'].append('201')
                # sync
                move("FORWARD")
                time.sleep(self.TIME_DELAY)
                shift, angle = self.detection(frame)
                last_shift, last_angle = self.getAction(shift, angle, last_shift, last_angle)
                self.flushFrame()
                break
            else:
                 continue
        idx = 0
        total_frame_num = 1
        frame_num = 0
        curr_address = ADDRESS[idx]
        frame_to_go = FRAME_INTERVAL[curr_address]

        while True:
            try:
                start = time.time()
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
                    frame = self.getFrame()
                    if self.frameQueue.qsize() > 1:
                        print('qsize:', self.frameQueue.qsize())
                        # self.flushFrame()

                    print('#', total_frame_num)
                    total_frame_num += 1
                    # hsv, stopFlag = detect_stop_sign(frame)
                    # self.sendFrame(hsv)
                    # if stopFlag:
                    #     curr_address = 'STOP'
                    #     idx = 0
                    #     next_address = ADDRESS[idx]
                    #     frame_to_go = FRAME_INTERVAL[next_address]
                    #     self.stopAddress(frame, curr_address, next_address)
                    # if frame_num == frame_to_go:
                    #     curr_address = ADDRESS[idx]
                    #     idx = (idx + 1) % NUM_ADDRESS
                    #     next_address = ADDRESS[idx]
                    #     frame_to_go = FRAME_INTERVAL[next_address]
                    #     self.stopAddress(frame, curr_address, next_address)
                    #     frame_num = 0
                    # else:
                    self.status['mode'] = 'move'
                    move("FORWARD")
                    shift, angle = self.detection(frame)
                    print(time.time() - start)
                    last_shift, last_angle = self.getAction(shift, angle, last_shift, last_angle)
                    frame_num += 1
            except:
                pass


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

    def detection(self, frame):
        frame_edge = edge_enhancement(frame)
        hsv = prepare_stop_pic(frame_edge)
        frame_line, shift, angle, obstacleFlag = lineDetect(frame, hsv)
        # frame,  shift, angle = lineDetect(frame, frame)

        # print('line time:', time.time() - start)
        # start = time.time()
        # detected = detect_sign(frame)
        # print('sign time:', time.time() - start)

        # if angle is None:
        #     move("STOP")

        # print(msg_t)
        # cv2.putText(frame, msg_t, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # send frame
        self.sendFrame(frame_line)
        return shift, angle


    def getAction(self, shift, angle,  last_shift, last_angle):

        leftSpeed = 0.5
        rightSpeed = 0.5
        rightCenter = conf.rightCenter
        rightAngle = conf.rightAngle

        if angle is not None:
            err = conf.shift_step * (shift - rightCenter) / 100 + conf.angle_step * (- (angle - rightAngle) / 90)
            # der = conf.shift_step *(shift - last_shift) / 100 + conf.angle_step * (angle - last_angle) / 180
            der = conf.shift_step * (shift - last_shift) / 100 + conf.angle_step * (-(angle - last_angle) / 90)
            PIDf = (err * conf.kp + der/4 * conf.kd) / 2
            logging.debug("linetrace | angle: %.5f, shift: %.5f, err:  %.5f, der = %.5f, PIDf:%.5f" % (angle, shift, err, der, PIDf))
            # print("PIDf:%.5f" % PIDf)
            # print("process time:", time.time()-startTime)
            if PIDf > 0:
                # print("RIGHT")
                motorSpeed(leftSpeed, rightSpeed - PIDf) #turn right
            else:
                # print("LEFT")
                motorSpeed(leftSpeed + 1.3 * PIDf, - rightSpeed)  #turn left
                # motorSpeed(-leftSpeed + PIDf, - rightSpeed)  #turn left

            last_shift = shift
            last_angle = angle
        else:
            logging.debug("linetrace | no line found")
            # move("BACKWARD")

        return last_shift, last_angle

    def stopMaintanence(self, curr_frame, location, path):
        move("STOP")
        self.status['mode'] = 'maintenance'
        if not self.frameQueue.empty():
            next_frame =self.getFrame()
        else:
            next_frame = curr_frame
        self.status['location'] = location
        # self.status['path'].append(path)
        time.sleep(1)
        self.flushFrame()
        self.frameQueue.put(curr_frame)
        # while True:
        #     if self.status['command'] == 'move':
        #         self.flushFrame()
        #         self.frameQueue.put(curr_frame)
        #         break
        #     else:
        #         continue

    def stopAddress(self, curr_frame, location, path):
        move("STOP")
        self.status['mode'] = 'stop'
        self.status['location'] = location
        print('current address:', location, 'next address:', path)
        if location == 'stop':
            self.status['path'] = []
        self.status['path'].append(path)
        time.sleep(1)
        self.flushFrame()
        self.frameQueue.put(curr_frame)
        # while True:
        #     if self.status['command'] == 'move':
        #         self.flushFrame()
        #         self.frameQueue.put(curr_frame)
        #         break
        #     else:
        #         continue

if __name__=='__main__':
    # log option
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    log_folder = '../log/standup3'
=======
    log_folder = './log'
>>>>>>> Stashed changes
=======
    log_folder = './log'
>>>>>>> Stashed changes
=======
    log_folder = './log'
>>>>>>> Stashed changes
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    timestr = time.strftime("/%Y%m%d_%H%M%S.log")
    logging.basicConfig(filename=log_folder + timestr, level=logging.DEBUG,
                        format='[%(asctime)s][%(levelname)s|%(threadName)s] >> %(message)s')

    frameQueue = queue.LifoQueue()
    threadLock = threading.Lock()
    status = {'command': 'move', 'mode': 'stop', 'location':'stop', 'path':[]}

    thread1 = ReadImageThread(threadLock, frameQueue)
    thread2 = CommunicateThread(threadLock, frameQueue, timeDelay= INIT_TIME_DELAY, showFrame = 'TRUE' , serverName='172.26.226.86', status=status)
    # thread3 = WebSocketThread(status=status)

    thread1.start()
    thread2.start()
    # thread3.start()

    thread1.join()
    thread2.join()
    # thread3.join()

