import os
import cv2
import numpy as np
import torch
from .read_data.default_boxes import CDefaultBoxes
from .utils.nms import py_cpu_nms
from .utils.box_utils import decode
root_dir = os.path.split(os.path.realpath(__file__))[0]


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
    return image


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for name, module1 in module._modules.items():
            recursion_change_bn(module1)
    return module


class TorchInference(object):
    def __init__(self, model_path=None, device=None):
        """
        对TorchInference进行初始化

        Parameters
        ----------
        model_path : str
            pytorch模型的路径，推荐使用绝对路径
        """
        super().__init__()
        self.model_path = model_path
        self.device = device
        if self.model_path is None:
            print("please set pytorch model path!\n")
            exit(-1)
        self.session = None
        self.model_loader()

    def model_loader(self):
        if torch.__version__ < "1.0.0":
            print("Pytorch version is not  1.0.0, please check it!")
            exit(-1)
        if self.model_path is None:
            print("Please set model path!!")
            exit(-1)
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # check_point = torch.load(self.checkpoint_file_path, map_location=self.device)
        # self.model = check_point['net'].to(self.device)
        self.session = torch.jit.load(self.model_path, map_location=self.device)
        # 如果模型为pytorch0.3.1版本，需要以下代码添加BN内的参数
        # for _, module in self.model._modules.items():
        #     recursion_change_bn(self.model)
        self.session.eval()

    def inference(self, x: torch.Tensor):
        """
        pytorch的推理
        Parameters
        ----------
        x : torch.Tensor
            pytorch模型输入

        Returns
        -------
        torch.Tensor
            pytorch模型推理结果
        """
        x = x.to(self.device)
        self.session = self.session.to(self.device)
        outputs = self.session(x)
        return outputs


class FaceDetector(TorchInference):
    def __init__(self, model_path=None, device=None):
        if model_path is None:
            model_path = os.path.join(root_dir, "models/mobilenet_v2_0.25_43_0.1162_jit.pth")
        super(FaceDetector, self).__init__(model_path, device)
        self.default_boxes = CDefaultBoxes().create_default_boxes()
        self.config = {
            'width': 160,
            'height': 160,
            'color_format': 'RGB',
            'mean': [0.486, 0.459, 0.408],
            'stddev': [0.229, 0.224, 0.225],
            'divisor': 255.0,
            'detect_threshold': 0.6,
            'nms_threshold': 0.3
        }

    def set_config(self, key, value):
        self.config[key] = value

    def _pre_process(self, image: np.ndarray) -> torch.Tensor:
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
        input_image = torch.from_numpy(input_image).float()
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

    def detect(self, input_image):
        input_image = self._pre_process(input_image)

        outputs = self.inference(input_image)
        conf_predict, loc_predict = outputs
        conf_predict = conf_predict.view(-1).sigmoid().data.cpu().numpy()
        loc_predict = loc_predict.view(-1, 4).data.cpu().numpy()
        loc_predict = decode(loc_predict, self.default_boxes)
        conf_predict = conf_predict.reshape(len(conf_predict), 1)
        boxes = np.hstack([loc_predict, conf_predict])

        boxes = self._post_process(boxes)
        return boxes


