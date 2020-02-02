import threading
import json
import numpy as np
import time

class FrameTransferThread(threading.Thread):
    def __init__(self, encodedFrameQueue, serverName='172.26.226.69', serverPort=8888):
        threading.Thread.__init__(self)
        self.encodedFrameQueue = encodedFrameQueue
        self.serverName = serverName
        self.serverPort = serverPort
        self.frameQueue = frameQueue

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
    thread1 = FrameTransferThread()
    thread1.start()
    thread1.join()