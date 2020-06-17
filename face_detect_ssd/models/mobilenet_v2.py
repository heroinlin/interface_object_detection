# -*- coding: utf-8 -*-

"""
code from: https://github.com/tonylins/pytorch-mobilenet-v2.git
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import math


def conv_bn(inp, oup, stride, padding=1):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(inp, oup, 3, stride, padding, bias=False)),
        ('bn', nn.BatchNorm2d(oup)),
        ('relu', nn.ReLU(inplace=True))
    ]))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)),
        ('bn', nn.BatchNorm2d(oup)),
        ('relu', nn.ReLU(inplace=True))
    ]))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = inp
        self.out_channels = oup
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.inverted_residual = nn.Sequential(OrderedDict([
            # pw
            ("conv1", nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False)),
            ("bn1", nn.BatchNorm2d(inp * expand_ratio)),
            ("relu1", nn.ReLU(inplace=True)),
            # dw
            ("conv2", nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False)),
            ("bn2", nn.BatchNorm2d(inp * expand_ratio)),
            ("relu2", nn.ReLU(inplace=True)),
            # pw-linear
            ("conv3", nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False)),
            ("bn3", nn.BatchNorm2d(oup)),
        ]))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inverted_residual(x)
        else:
            return self.inverted_residual(x)


class CMobileNetV2(nn.Module):
    def __init__(self, width_multi=0.25):
        super(CMobileNetV2, self).__init__()
        self.classes_number = 1

        # setting of inverted residual blocks
        self.inverted_residual_setting = [
            # t, c,  n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_multi)
        base_features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = int(c * width_multi)
            for i in range(n):
                if i == 0:
                    base_features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    base_features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.base_features = nn.Sequential(*base_features)

        extra_features = list()
        extra_features.append(conv_1x1_bn(round(320*width_multi), round(256*width_multi)))
        extra_features.append(conv_bn(round(256*width_multi), round(512*width_multi), 2))
        extra_features.append(conv_1x1_bn(round(512*width_multi), round(128*width_multi)))
        # extra_features.append(conv_bn(round(128*width_multi), round(256*width_multi), 2))
        # extra_features.append(conv_1x1_bn(round(256*width_multi), round(128*width_multi)))
        extra_features.append(conv_bn(round(128*width_multi), round(256*width_multi), 1, padding=0))
        self.extra_features = nn.Sequential(*extra_features)

        self.conf_layers = []
        self.conf_layers.append(nn.Conv2d(round(32*width_multi),   1 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(round(96*width_multi),   3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(round(320*width_multi),  3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(round(512*width_multi),  3 * self.classes_number, 3, 1, 1))
        # self.conf_layers.append(nn.Conv2d(round(256*width_multi),  3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(round(256*width_multi),  3 * self.classes_number, 1, 1, 0))
        self.conf_model = nn.ModuleList(self.conf_layers)

        self.loc_layers = []
        self.loc_layers.append(nn.Conv2d(round(32*width_multi),   1 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(round(96*width_multi),   3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(round(320*width_multi),  3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(round(512*width_multi),  3 * 4, 3, 1, 1))
        # self.loc_layers.append(nn.Conv2d(round(256*width_multi),  3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(round(256*width_multi),  3 * 4, 1, 1, 0))
        self.loc_model = nn.ModuleList(self.loc_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        modules = list(self.modules())
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        base_feature_indices = [6, 13, 17]
        # base_feature_indices = [13, 17]
        # extra_feature_indices = [1, 3, 5]
        extra_feature_indices = [1, 3]
        conf, loc = list(), list()
        for i, feature in enumerate(self.base_features):
            x = feature(x)
            # for k, (name1, module) in enumerate(feature._modules.items()):
            #     name1 =(i, name1)
            #     if isinstance(module, nn.Sequential):
            #         for k, (name2, module1) in enumerate(module._modules.items()):
            #             name2 = name1.__add__((name2,))
            #             x = module1(x)
            #             print(name2, module1)
            #     else:
            #         x = module(x)
            #         print(name1, module)
            if i in base_feature_indices:
                # print(x.size())
                index = base_feature_indices.index(i)
                conf_predict = self.conf_model[index](x).permute(0, 2, 3, 1).contiguous()
                loc_predict = self.loc_model[index](x).permute(0, 2, 3, 1).contiguous()
                conf.append(conf_predict.view(conf_predict.size(0), -1, self.classes_number))
                loc.append(loc_predict.view(loc_predict.size(0), -1, 4))
        for i, feature in enumerate(self.extra_features):
            x = feature(x)
            if i in extra_feature_indices:
                # print(x.size())
                index = extra_feature_indices.index(i) + 3
                conf_predict = self.conf_model[index](x).permute(0, 2, 3, 1).contiguous()
                loc_predict = self.loc_model[index](x).permute(0, 2, 3, 1).contiguous()
                conf.append(conf_predict.view(conf_predict.size(0), -1, self.classes_number))
                loc.append(loc_predict.view(loc_predict.size(0), -1, 4))
        conf = torch.cat(conf, dim=1)
        loc = torch.cat(loc, dim=1)
        return conf, loc


def main():
    # load pre trained model

    # check_point_file_path = r"F:\models\pytorch\mobilenetv2_718.pth.tar"
    # check_point = torch.load(check_point_file_path)
    # check_point_new = OrderedDict()
    # for name, v in check_point.items():
    #     check_point_new[name.replace('module.features', 'base_features')] = v
    #
    # model = CMobileNetV2()
    # model.load_state_dict(check_point_new, strict=False)

    model = CMobileNetV2()
    input_x = torch.autograd.Variable(torch.randn(16, 3, 160, 160))
    # print(model)
    conf, loc = model(input_x)
    print(conf.size(), loc.size())


if __name__ == "__main__":
    main()
