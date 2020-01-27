import cv2
import struct
import numpy as np
import time
from socket import *
from linedetect import lineDetect

# Opens server
serverPort = 8888
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(('', serverPort))
serverSocket.listen(1)

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
    
    try:
        previousTime = time.time()
        i=0
        while True:
            # Get message
            length = recvall(connectionSocket, 16)
            print('image size:', int(length), 'bytes')
            stringData = recvall(connectionSocket, int(length))
            frame = cv2.imdecode(np.fromstring(stringData, dtype='uint8'), 1)
            cv2.imwrite("../demo/track/demo/Img_%d.jpg"%i, frame)
            i+=1
            # frame, position, angle = lineTrack(frame)
            # See image
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

            currentTime = time.time()
            intervals.append(currentTime - previousTime)
            previousTime = currentTime

        intervalArray = np.array(intervals)
        print('average:', np.mean(intervalArray))
        print('max:', np.max(intervalArray))
        print('min:', np.min(intervalArray))
        print(np.sort(intervalArray)[:-10:-1])
    except:
        print("Waiting for connection...")