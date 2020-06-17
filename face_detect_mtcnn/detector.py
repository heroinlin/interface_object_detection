from __future__ import print_function

import math

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from .models import PNet, RNet, ONet
from .utils import nms, convert_to_square, calibrate_box, correct_bboxes


class FaceDetector(object):
    def __init__(self, checkpoint_file_path=None, detect_threshold=0.7, nms_threshold=0.3, device=None):
        super(FaceDetector, self).__init__()
        self.checkpoint_file_path = checkpoint_file_path
        self.default_boxes = None
        self.transforms = None
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.device = device
        self.config = {
            "width": 160,
            "height": 144,
            "mean": [0.446, 0.446, 0.443],
            "stddev": [0.248, 0.246, 0.249],
            "detect_threshold": detect_threshold,
            "nms_threshold": nms_threshold
        }

    def _preprocess(self, img):
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = (img - 127.5) * 0.0078125
        return img

    def _generate_bboxes(self, probs, offsets, scale, threshold):
        stride = 2
        cell_size = 12

        inds = np.where(probs > threshold)
        if inds[0].size == 0:
            return np.array([])

        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        bounding_boxes = np.vstack([
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale), score, offsets
        ])
        return bounding_boxes.T

    def get_image_boxes(self, bounding_boxes, img, size=24):
        num_boxes = len(bounding_boxes)
        width = img.shape[1]
        height = img.shape[0]

        [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(
            bounding_boxes, width, height)
        img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

        for i in range(num_boxes):
            img_box = np.zeros((h[i], w[i], 3), 'uint8')

            img_array = np.asarray(img, 'uint8')
            img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
                img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

            img_box = cv2.resize(img_box, (size, size))
            img_box = np.asarray(img_box, 'float32')
            img_boxes[i, :, :, :] = self._preprocess(img_box)
        return img_boxes

    def run_first_stage(self, image, net, scale, threshold):
        height, width = image.shape[:2]
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = cv2.resize(image, (sw, sh))
        img = np.asarray(img, 'float32')
        img = Variable(torch.FloatTensor(self._preprocess(img)), volatile=True)
        output = net(img)
        probs = output[1].data.numpy()[0, 1, :, :]
        offsets = output[0].data.numpy()
        boxes = self._generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None
        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]

    def detect(self, image,
                     min_face_size=20.0,
                     thresholds=[0.6, 0.7, 0.8],
                     nms_thresholds=[0.7, 0.7, 0.7]):
        self.onet.eval()

        height, width = image.shape[:2]
        min_length = min(height, width)
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []
        m = min_detection_size / min_face_size
        min_length *= m
        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1
        bounding_boxes = []
        for s in scales:  # run P-Net on different scales
            boxes = self.run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        if not len(bounding_boxes):
            return [], []
        bounding_boxes = np.vstack(bounding_boxes)
        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5],
                                       bounding_boxes[:, 5:])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2
        img_boxes = self.get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = self.rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3
        img_boxes = self.get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = self.onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(
            xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(
            ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks

# if __name__ == '__main__':
#     import cv2
#     face_detector = FaceDetector()
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, img = cap.read()
#         if not ret: break
#         bounding_boxes, landmarks = detect(img)
#         image = show_bboxes(img, bounding_boxes, landmarks)
#
#         # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         cv2.imshow('0', image)
#         if cv2.waitKey(10) == 27:
#             break
