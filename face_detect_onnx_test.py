import cv2
import os
import sys
working_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(working_root)
from face_detect_onnx import Detector, draw_detection_rects
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})


class ObjectDetect(object):
    def __init__(self):
        self.detector = Detector()

    def detect(self, image):
        return self.detector.detect(image)


if __name__ == '__main__':
    object_detector = ObjectDetect()
    root_dir = os.path.split(os.path.realpath(__file__))[0]
    image_path = os.path.dirname(os.path.dirname(root_dir)) + "/data/image_data/images/00000_1.jpg"
    image = cv2.imread(image_path)
    boxes = object_detector.detect(image)
    print(boxes)
    draw_detection_rects(image, boxes)
    cv2.imshow("detect", image)
    cv2.waitKey()
