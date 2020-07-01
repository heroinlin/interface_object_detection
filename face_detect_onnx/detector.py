# -*- coding: utf-8 -*-

import numpy as np
import onnxruntime

import os
import cv2

from .defaults import _C as cfg
from .utils import py_cpu_nms, change_box_order

working_root = os.path.split(os.path.realpath(__file__))[0]


class ONNXInference(object):
    def __init__(self, onnx_file_path=None):
        """
        对ONNXInference进行初始化

        Parameters
        ----------
        onnx_file_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        super().__init__()
        self.onnx_file_path = onnx_file_path
        if self.onnx_file_path is None:
            print("please set onnx model path!\n")
            exit(-1)
        self.session = onnxruntime.InferenceSession(self.onnx_file_path)

    def inference(self, x: np.ndarray):
        """
        onnx的推理
        Parameters
        ----------
        x : np.ndarray
            onnx模型输入

        Returns
        -------
        np.ndarray
            onnx模型推理结果
        """
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run(output_names=[output_name],
                                   input_feed={input_name: x.astype(np.float32)})
        return outputs


class Detector(ONNXInference):
    def __init__(self, onnx_file_path=None):
        """对Detector进行初始化

        Parameters
        ----------
        onnx_file_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        if onnx_file_path is None:
            onnx_file_path = os.path.join(working_root,
                                          'onnx_model',
                                          "mobilenet_v2_184_0.1701-sim.onnx")
        super(Detector, self).__init__(onnx_file_path)
        self.cfg = cfg.clone()
        self.cfg.freeze()

        self.obj_threshold = cfg.INPUT.OBJ_THRESHOLD
        self.nms_threshold = cfg.INPUT.NMS_THRESHOLD

    def _pre_process(self, image: np.ndarray) -> np.ndarray:
        """对图像进行预处理

        Parameters
        ----------
        image : np.ndarray
            输入的原始图像，BGR格式，通常使用cv2.imread读取得到

        Returns
        -------
        np.ndarray
            原始图像经过预处理后得到的数组
        """
        if self.cfg.INPUT.FORMAT == "RGB":
            image = image[:, :, ::-1]
        image = cv2.resize(image, (cfg.INPUT.WIDTH, cfg.INPUT.HEIGHT))
        input_image = (np.array(image, dtype=np.float32) / 255 - cfg.INPUT.PIXEL_MEAN) / cfg.INPUT.PIXEL_STD
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, 0)
        return input_image

    def _post_process(self, boxes: np.ndarray) -> np.ndarray:
        """
        对网络输出框进行后处理
        Parameters
        ----------
        boxes: np.ndarray
            网络输出框
        Returns
        -------
            np.ndarray
            返回值维度为(n, 5)，其中n表示目标数量，5表示(x1, y1, x2, y2, score)
        """
        indices = np.where(boxes[:, 4] > self.obj_threshold)
        boxes = boxes[indices]
        # boxes = change_box_order(boxes, order="xywh2xyxy")
        keep = py_cpu_nms(boxes, self.nms_threshold)
        boxes = boxes[keep]
        return boxes

    def detect(self, image: np.ndarray) -> np.ndarray:
        """检测前门图片中乘客目标

        Parameters
        ----------
        image : np.ndarray
            输入图片，BGR格式，通常使用cv2.imread获取得到

        Returns
        -------
        np.ndarray
            返回值维度为(n, 5)，其中n表示目标数量，5表示(x1, y1, x2, y2, score)
        """
        image = self._pre_process(image)
        outputs = self.inference(image)
        boxes = outputs[0].squeeze()
        boxes = self._post_process(boxes)
        return np.array(boxes)
