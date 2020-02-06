from multiprocessing import Process, Queue
import cv2

class ReadImageProcess(Process):
    def __init__(self, frameQueue, width=320, height=240, frameRate=20, bufferSize=1):
        super(ReadImageProcess, self).__init__()
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
        while True:
            ret, frame = self.cap.read()
            # ret, encodedFrame = cv2.imencode('.jpg', frame, self.encodeParam)
            if self.frameQueue.qsize() < 10:
                self.frameQueue.put(frame)
            else:
                print('qsize >= 10')
            

if __name__=='__main__':
    frameQueue = Queue()
    process1 = ReadImageProcess(frameQueue)
    process1.daemon = True
    process1.start()
    process1.join()