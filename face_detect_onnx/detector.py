# -*- coding: utf-8 -*-

import numpy as np
import onnxruntime

import os
import cv2

from .utils import py_cpu_nms

working_root = os.path.split(os.path.realpath(__file__))[0]


class ONNXInference(object):
    def __init__(self, model_path=None):
        """
        对ONNXInference进行初始化

        Parameters
        ----------
        onnx_file_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        super().__init__()
        self.model_path = model_path
        if self.model_path is None:
            print("please set onnx model path!\n")
            exit(-1)
        self.session = onnxruntime.InferenceSession(self.model_path)

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
    def __init__(self, model_path=None):
        """对Detector进行初始化

        Parameters
        ----------
        onnx_file_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        if model_path is None:
            model_path = os.path.join(working_root,
                                          'onnx_model',
                                          "mobilenet_v2_0.25_43_0.1162-sim.onnx")
        super(Detector, self).__init__(model_path)
        self.config = {
            'width': 160,
            'height': 144,
            'color_format': 'RGB',
            'mean': [0.4914, 0.4822, 0.4465],
            'stddev': [0.247, 0.243, 0.261],
            'divisor': 255.0,
            'detect_threshold': 0.6,
            'nms_threshold': 0.3
        }

    def set_config(self, key, value):
        self.config[key] = value

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
        if self.config['color_format'] == "RGB":
            image = image[:, :, ::-1]
        if self.config['width'] > 0 and self.config['height'] > 0:
            image = cv2.resize(image, (self.config['width'], self.config['height']))
        input_image = (np.array(image, dtype=np.float32) / self.config['divisor'] - self.config['mean']) / self.config['stddev']
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
        indices = np.where(boxes[:, 4] > self.config['detect_threshold'])
        boxes = boxes[indices]
        # boxes = change_box_order(boxes, order="xywh2xyxy")
        keep = py_cpu_nms(boxes, self.config['nms_threshold'])
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
