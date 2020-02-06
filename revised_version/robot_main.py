from multiprocessing import Queue, Manager
import numpy
from websocket_communicate import WebsocketCommunicateProcess
from read_image import ReadImageProcess
from image_processing import ImageProcessingProcess
from frame_transfer import FrameTransferProcess
from line_detect.movement import *

if __name__ == '__main__':
    frameQueue = Queue()
    sendQueue = Queue()
    manager = Manager()
    status = manager.dict({'command':'empty', 'path':''})
    websocketProcess = WebsocketCommunicateProcess(uri="ws://172.26.226.69:3000/robot", status=status)
    readImageProcess = ReadImageProcess(frameQueue=frameQueue, width=320, height=240, frameRate=20)
    frameTransferProcess = FrameTransferProcess(encodedFrameQueue=sendQueue, serverName='172.26.226.69')
    imageProcessingProcess = ImageProcessingProcess(frameQueue=frameQueue, sendQueue=sendQueue, status=status)

    try:
        websocketProcess.daemon = True
        readImageProcess.daemon = True
        frameTransferProcess.daemon = True
        imageProcessingProcess.daemon = True

        websocketProcess.start()
        readImageProcess.start()
        frameTransferProcess.start()
        imageProcessingProcess.start()

        websocketProcess.join()
        readImageProcess.join()
        frameTransferProcess.join()
        imageProcessingProcess.join()
    except KeyboardInterrupt:
        move('STOP')