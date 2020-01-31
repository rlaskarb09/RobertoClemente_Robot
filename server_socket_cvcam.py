import cv2
import struct
import numpy as np
import sys
import os
import time
from socket import *
from num_detect.detect_stop_addr_function import *

# frame save folder
parent_dir = '/Users/chwaaaa/Downloads/CMU/Project/demo/0131'
if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)

# Opens server
serverPort = 8888
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(('', serverPort))
serverSocket.listen(1)
ACTION_LENGTH = 11

print("The server is ready to receive on port", serverPort)

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

while True:
    # Accepts connection
    (connectionSocket, clientAddress) = serverSocket.accept()
    print('Connection requested from', clientAddress)

    intervals = []
    pauseFlag = -1
    try:
        previousTime = time.time()
        i = 0
        while True:
            # Get message
            length = recvall(connectionSocket, 16)
            print('image size:', int(length), 'bytes')
            stringData = recvall(connectionSocket, int(length))

            frame = cv2.imdecode(np.fromstring(stringData, dtype='uint8'), 1)
            cv2.imwrite(parent_dir + "/img_%d.jpg" % i, frame)
            i += 1
            # See image
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key > -1:
                pauseFlag = pauseFlag*(-1)
            if pauseFlag  == 1:
                actionMessage = 'KEY_PRESSED'
                connectionSocket.send(actionMessage.encode().ljust(ACTION_LENGTH))
            else:
                actionMessage = 'NO_KEY'
                connectionSocket.send(actionMessage.encode().ljust(ACTION_LENGTH))

            currentTime = time.time()
            intervals.append(currentTime - previousTime)
            previousTime = currentTime

        intervalArray = np.array(intervals)
        print('average:', np.mean(intervalArray))
        print('max:', np.max(intervalArray))
        print('min:', np.min(intervalArray))
        print(np.sort(intervalArray)[:-10:-1])

    except KeyboardInterrupt:
        sys.exit()
    except:
        # cv2.destroyAllWindows()
        print("Waiting for connection...")
    finally:
        cv2.destroyAllWindows()

