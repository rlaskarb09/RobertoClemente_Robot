from multiprocessing import Process, Queue
import json
import numpy as np
import time
from socket import *

class FrameTransferProcess(Process):
    def __init__(self, encodedFrameQueue, serverName='172.26.225.69', serverPort=8888):
        super(FrameTransferProcess, self).__init__()
        self.encodedFrameQueue = encodedFrameQueue
        self.serverName = serverName
        self.serverPort = serverPort

    def run(self):
        try:
            # clientSocket
            self.clientSocket = socket(AF_INET, SOCK_STREAM)
            self.clientSocket.connect((self.serverName, self.serverPort))
            print('connected with the image processing server.')
            while True:
                if not self.encodedFrameQueue.empty():
                    encodedFrame = self.encodedFrameQueue.get()
                    encodedString = encodedFrame.tostring()
                    self.clientSocket.send(str(len(encodedString)).encode().ljust(16))
                    self.clientSocket.send(encodedString)
                # time.sleep(0.0001)

        except KeyboardInterrupt:
            print('exception from FrameTransferThread')

if __name__=='__main__':
    sendQueue = Queue()
    process1 = FrameTransferProcess(sendQueue)
    process1.daemon = True
    process1.start()
    process1.join()