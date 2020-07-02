import torch
import models
import sys
import os
from collections import OrderedDict
import math
import numpy as np
from itertools import product
from combine_conv_bn import fuse_module

cfg_ssd = {
        'steps': [8, 16, 32, 64, 128],
        'clip': True,
        'image_size': [144, 160],
        's_min': 0.15,
        's_max': 0.9,
        'aspect_ratios': [[1], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]]
    }


def decode(loc, priors_xy, priors_wh):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes_xy, boxes_wh = torch.split(loc, 2, dim=2)  # 转换为Slice操作ncnn20191113、20200426版本均可用
    # boxes_xy = loc[..., 0:2]  # 转换为Split+Crop操作, 此写法ncnn20191113版本支持，20200426版本Crop失效不起切片作用
    # boxes_wh = loc[..., 2:4]
    boxes = torch.cat((
        priors_xy + boxes_xy * priors_wh,
        priors_wh * torch.exp(boxes_wh)), 2)
    return boxes

def transform_box(boxes):
    boxes_xy, boxes_wh = torch.split(boxes, 2, dim=2)
    bboxes = torch.cat((boxes_xy - boxes_wh/2, boxes_xy + boxes_wh/2), 2)
    # bboxes = torch.clamp(bboxes, 0, 1.0)
    return bboxes

class SSDPriorBox(object):
    def __init__(self, cfg):
        self.image_size = cfg['image_size']  # [height, width]
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.s_min = cfg['s_min']
        self.s_max = cfg['s_max']
        self.clip = cfg['clip']
        self.feature_maps = [[math.ceil(self.image_size[0] / step), math.ceil(self.image_size[1] / step)]
                             for step in self.steps]
        self.feature_maps[-1] = [1, 1]
        self._scales = list()
        self._create_scales()

        assert len(self.feature_maps) == len(self._scales)
        assert len(self.steps) == len(self._scales)
        assert len(self.aspect_ratios) == len(self._scales)

    def _create_scales(self):
        min_scale = self.s_min
        max_scale = self.s_max
        feature_maps_count = len(self.feature_maps)
        for i in range(feature_maps_count):
            scale = min_scale + (max_scale - min_scale) * i / (feature_maps_count - 1)
            self._scales.append(scale)

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            assert len(f) == 2
            for y, x in product(list(range(f[0])), list(range(f[1]))):
                # corresponding coordinates in input image
                cx = (x + 0.5) * self.steps[k] / self.image_size[1]
                cy = (y + 0.5) * self.steps[k] / self.image_size[0]
                for ar in self.aspect_ratios[k]:
                    anchors += [cx, cy, self._scales[k] * math.sqrt(ar), self._scales[k] / math.sqrt(ar)]
        # back to torch land
        output = torch.Tensor(anchors).view(1, -1, 4)
        if self.clip:
            output = np.clip(output, a_min=0, a_max=1)
        return output


class SSDHead(torch.nn.Module):
    def __init__(self, model):
        super(SSDHead, self).__init__()
        self.device =  f"cuda:0" if torch.cuda.is_available() else "cpu"
        priors = SSDPriorBox(cfg_ssd)
        # image_size = torch.tensor([160, 144, 160, 144], dtype=torch.float32).to(self.device)
        # self.image_size = image_size.repeat((735, 1)).unsqueeze(0)
        self.model = model
        self.decode = decode
        self.transform_box = transform_box
        self.priors = priors.forward()
        self.priors = self.priors.to(self.device)
        self.priors_xy = self.priors[..., 0:2]
        self.priors_wh = self.priors[..., 2:4]

    def forward(self, x):
        score, boxes = self.model(x)
        boxes = decode(boxes, self.priors_xy, self.priors_wh)
        score = torch.sigmoid(score)
        bboxes = self.transform_box(boxes)
        output = torch.cat([bboxes, score], 2)
        # boxes = boxes * self.image_size
        return output


def export_model(checkpoint_path, export_model_name, inputsize=[1, 3, 144, 160]):
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    check_point = torch.load(checkpoint_path, map_location=device)
    model = check_point['net']
    state_dict = check_point['net'].state_dict()
    # model = models.init_model(name="resnet18mid", pretrained=False, num_classes=9991)
    # state_dict = check_point['net']
    model = model.to(device) if device == f"cuda:0" else model
    model.load_state_dict(state_dict)
    model.eval()
    ssd_head = SSDHead(model)
    fuse_module(ssd_head)
    for key, value in state_dict.items():
        print(key)
    dummy_input = torch.randn(inputsize).to(device)
    # # torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True)
    torch.onnx.export(model=ssd_head, args=dummy_input, f=export_model_name, verbose=True, input_names=['image'],
                      output_names=['outTensor'])  # 0.4.0以上支持更改输入输出层名称


def onnx_simplifier(checkpoint_path, export_model_name):
    import onnx
    from onnxsim import simplify

    # load your predefined ONNX model
    model = onnx.load(checkpoint_path)

    # convert model
    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, export_model_name)


if __name__ == '__main__':
    root_dir = os.path.split(os.path.realpath(__file__))[0]
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(root_dir)),\
        r"face_detect_ssd\models\mobilenet_v2_0.25_43_0.1162.pth")
    export_model_name = checkpoint_path.replace(".pth", ".onnx")
    sim_export_model_name = checkpoint_path.replace(".pth", "-sim.onnx")
    export_model(checkpoint_path, export_model_name)
    onnx_simplifier(export_model_name, sim_export_model_name)
