import cv2
import time
import os
import sys
import importlib
import time
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_PATH)
sys.path.append(BASE_PATH)
importlib.reload(sys)
from YoloV5Detector.V5Detector import Detector



if __name__ == '__main__':
    weights_path = "weights/5.0/yolov5s.pt"
    image_path = "samples/images/zidane.jpg"

    det = Detector(weights_path)
    cls = ['person', 'car']
    thresh = 0.3
    for i in range(100):
        t1 = time.time()
        img = cv2.imread(image_path)
        img_res, det_res = det.detect(img, cls, thresh)
        t2 = (time.time() - t1) * 1000
        print("time:{} ms, res:{}".format(t2, det_res))


