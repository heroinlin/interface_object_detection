# -*- coding: utf-8 -*-
import cv2
import numpy as np


def change_box_order(boxes, order):
    """
    change box order between (xmin, ymin, xmax, ymax) and (center_x, center_y, width, height)

    Args:
        boxes: (numpy array), bounding boxes, size [N, 4]
        order: (str), 'xyxy2xywh' or 'xywh2xyxy'
    Returns:
        converted bounding boxes, size [N, 4]
    """
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    if order == 'xyxy2xywh':
        boxes[:, :2] = (boxes[:, :2] + boxes[:, 2:4]) / 2
        boxes[:, 2:4] -= boxes[:, :2]
    else:
        boxes[:, :2] -= boxes[:, 2:4] / 2
        boxes[:, 2:4] += boxes[:, :2]
    return boxes


def py_cpu_nms(boxes, nms_thresh=0.3):
    """Pure Python NMS baseline."""
    scores = boxes[:, 4]
    bboxes = boxes[:, 0:4]
    # bboxes = change_box_order(bboxes, order="xywh2xyxy")
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        indices = np.where(ovr <= nms_thresh)[0]
        order = order[indices + 1]
    return keep


def box_transform(bounding_boxes, width, height):
    """
    bounding_boxes  [[score, box],[score, box]]
    box框结果值域由[0,1],[0,1] 转化为[0,width]和[0,height]
    """
    for i in range(len(bounding_boxes)):
        x1 = float(bounding_boxes[i][1])
        y1 = float(bounding_boxes[i][2])
        x2 = float(bounding_boxes[i][3])
        y2 = float(bounding_boxes[i][4])
        bounding_boxes[i][1] = x1 * width
        bounding_boxes[i][2] = y1 * height
        bounding_boxes[i][3] = x2 * width
        bounding_boxes[i][4] = y2 * height
    return bounding_boxes


def draw_detection_rects(image: np.ndarray, detection_rects: np.ndarray, color=(0, 255, 0), method=1):
    if not isinstance(detection_rects, np.ndarray):
        detection_rects = np.array(detection_rects)
    if method:
        width = image.shape[1]
        height = image.shape[0]
    else:
        width = 1.0
        height = 1.0
    for index in range(detection_rects.shape[0]):
        cv2.rectangle(image,
                      (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                      (int(detection_rects[index, 2] * width), int(detection_rects[index, 3] * height)),
                      color,
                      thickness=2)
        if detection_rects.shape[1] == 5:
            cv2.putText(image, f"{detection_rects[index, 4]:.03f}",
                        (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                        1, 1, (255, 0, 255))


def main():
    boxes = np.array([[0.54744893, 0.1931118, 0.74389774, 0.40391764],
                      [0.56696707, 0.18375029, 0.7563599, 0.392475],
                      [0.5336509, 0.19298509, 0.742272, 0.41128263],
                      [0.55161643, 0.20036203, 0.7525438, 0.40939254],
                      [0.53546613, 0.22605795, 0.73250383, 0.42756397],
                      [0.56147254, 0.22586273, 0.7517569, 0.4337405],
                      [0.3415279, 0.77885133, 0.5409234, 0.9972338],
                      [0.3729902, 0.77662545, 0.5647705, 0.98679537],
                      [0.33567593, 0.79856724, 0.5487639, 1.0065217],
                      [0.3603476, 0.7869965, 0.56771946, 0.9889393],
                      [0.07845123, 0.11481889, 0.3822456, 0.42998838],
                      [0.0842551, 0.14087763, 0.37125608, 0.43346557],
                      [0.51495427, 0.10305066, 0.77949303, 0.41245198],
                      [0.5210506, 0.13213347, 0.77125347, 0.4289379],
                      [0.10251364, 0.15086217, 0.38026914, 0.43821383],
                      [0.07757787, 0.14891642, 0.369264, 0.453403],
                      [0.11547841, 0.14710297, 0.39173675, 0.43716407],
                      [0.07686087, 0.14307268, 0.37073565, 0.44173372],
                      [0.0780358, 0.14480156, 0.37374604, 0.44701266],
                      [0.07763492, 0.1436856, 0.37257123, 0.44222152],
                      [0.08486634, 0.12701502, 0.38850933, 0.43897173],
                      [0.077886, 0.13691053, 0.37450182, 0.43408194],
                      [0.50535625, 0.15558976, 0.7573026, 0.43900514],
                      [0.5110434, 0.15676989, 0.7668512, 0.4350286],
                      [0.51693, 0.15033199, 0.7559277, 0.43256253],
                      [0.5164166, 0.14777605, 0.7607021, 0.44244152],
                      [0.511532, 0.15290423, 0.75861186, 0.42668283],
                      [0.5318208, 0.1355991, 0.78191686, 0.4352294],
                      [0.5122352, 0.14042331, 0.7644748, 0.4246599],
                      [0.07975, 0.1513847, 0.36155456, 0.47526944],
                      [0.07333057, 0.1443876, 0.36208856, 0.4757933],
                      [0.08328708, 0.1348223, 0.38037848, 0.4753334],
                      [0.08223023, 0.13542484, 0.36910868, 0.4444744],
                      [0.08236438, 0.12999213, 0.36838257, 0.4307695],
                      [0.07298808, 0.13934608, 0.3756444, 0.46980745],
                      [0.5165012, 0.15055762, 0.755497, 0.45122337],
                      [0.51842207, 0.14345999, 0.756038, 0.43936002],
                      [0.31804347, 0.6708397, 0.5884042, 0.99870735],
                      [0.3253979, 0.7048567, 0.57989824, 1.0095061],
                      [0.32156897, 0.7017491, 0.5776139, 0.98845553],
                      [0.32798213, 0.7078333, 0.58452046, 0.9906529],
                      [0.32463565, 0.7167139, 0.5681728, 0.9950621],
                      [0.32523614, 0.7123416, 0.56885576, 0.99501616],
                      [0.3204942, 0.71075034, 0.5719048, 0.99109876],
                      [0.32402343, 0.6984119, 0.5853929, 0.99281985],
                      [0.31291234, 0.70549977, 0.57003355, 0.9886979],
                      [0.31672662, 0.69927263, 0.57962304, 0.9733825],
                      [0.3230359, 0.6833366, 0.57670265, 0.96512496]])
    scores = np.array([0.4294565, 0.46359357, 0.59797174, 0.60733914, 0.42092034, 0.4815626,
                       0.39068827, 0.48891723, 0.40007874, 0.46394715, 0.51119643, 0.72646916,
                       0.44761348, 0.63818437, 0.8223701, 0.52652436, 0.84188104, 0.9130251,
                       0.8543714, 0.8232446, 0.64431524, 0.7468325, 0.66855913, 0.66801983,
                       0.8387958, 0.7829686, 0.7029136, 0.5126272, 0.581237, 0.45108745,
                       0.42674273, 0.40537232, 0.72780097, 0.8265414, 0.4635221, 0.6402197,
                       0.7699671, 0.48072517, 0.6767659, 0.66149956, 0.72994214, 0.89210844,
                       0.8460373, 0.7117947, 0.6816967, 0.72382104, 0.69690615, 0.8139456])
    keep = py_cpu_nms(boxes, scores, 0.5)
    # print(keep)
    boxes = boxes[keep]
    scores = scores[keep]
    print(boxes, scores)


if __name__ == "__main__":
    main()
