import cv2
import threading
import time

class Camera:
    def __init__(self, src=0,shape=[480,320]):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, shape[0])
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, shape[1])
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter::fourcc('M', 'J', 'P', 'G'));
        # self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'));
        print("sss")
        self.stopped = False
        for _ in range(10): #warm up the camera
            (self.grabbed, self.frame) = self.stream.read()
        

    def start(self):
        threading.Thread(target=self.update, args=()).start()

    def update(self):
        count = 0
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            # time.sleep(1)
            if self.src == 0:
                path = "images/{}.png".format(count)
                count = count + 1
                cv2.imwrite(path, self.frame)


    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True