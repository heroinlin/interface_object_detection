import math
import torch
import torch.nn as nn
import torch.nn.init as init


class NU_Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, expand5x5_planes, expand7x7_planes):
        super(NU_Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_batch_norm = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_batch_norm = nn.BatchNorm2d(expand1x1_planes)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_batch_norm = nn.BatchNorm2d(expand3x3_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.expand5x5 = nn.Conv2d(squeeze_planes, expand5x5_planes,
                                   kernel_size=5, padding=2)
        self.expand5x5_batch_norm = nn.BatchNorm2d(expand5x5_planes)
        self.expand5x5_activation = nn.ReLU(inplace=True)
        self.expand7x7 = nn.Conv2d(squeeze_planes, expand7x7_planes,
                                   kernel_size=7, padding=3)
        self.expand7x7_batch_norm = nn.BatchNorm2d(expand7x7_planes)
        self.expand7x7_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_batch_norm(x)
        x = self.squeeze_activation(x)
        return torch.cat([
            self.expand1x1_activation(self.expand1x1_batch_norm(self.expand1x1(x))),
            self.expand3x3_activation(self.expand3x3_batch_norm(self.expand3x3(x))),
            self.expand5x5_activation(self.expand5x5_batch_norm(self.expand5x5(x))),
            self.expand7x7_activation(self.expand7x7_batch_norm(self.expand7x7(x))),
        ], 1)


class NU_SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(NU_SqueezeNet, self).__init__()
        if version not in [1.0, 1.1, 1.2]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 or 1.2 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                NU_Fire(96, 16, 32, 32, 32, 32),
                NU_Fire(128, 16, 32, 32, 32, 32),
                NU_Fire(128, 32, 64, 64, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                NU_Fire(256, 32, 64, 64, 64, 64),
                NU_Fire(256, 48, 96, 96, 96, 96),
                NU_Fire(384, 48, 96, 96, 96, 96),
                NU_Fire(384, 64, 128, 128, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                NU_Fire(512, 64, 128, 128, 128, 128),
            )
        elif version == 1.1:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                NU_Fire(64, 16, 32, 32, 32, 32),
                NU_Fire(128, 16, 32, 32, 32, 32),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                NU_Fire(128, 32, 64, 64, 64, 64),
                NU_Fire(256, 32, 64, 64, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                NU_Fire(256, 48, 96, 96, 96, 96),
                NU_Fire(384, 48, 96, 96, 96, 96),
                NU_Fire(384, 48, 96, 96, 96, 96),
                NU_Fire(512, 48, 96, 96, 96, 96),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.Conv2d(64, 64, kernel_size=1, stride=2,  bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                NU_Fire(64, 16, 32, 32, 32, 32),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                NU_Fire(128, 32, 64, 64, 64, 64),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        final_avgpool = nn.AvgPool2d(13, stride=1)
        if version == 1.2:
            final_conv = nn.Conv2d(256, self.num_classes, kernel_size=1)
            final_avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            final_avgpool
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)
        # return x


def NU_squeezeNet1_0(**kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = NU_SqueezeNet(version=1.0, **kwargs)
    return model


def NU_squeezeNet1_1(**kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = NU_SqueezeNet(version=1.1, **kwargs)
    return model


def NU_squeezeNet1_2(**kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = NU_SqueezeNet(version=1.2, **kwargs)
    return model


class NetCrop(nn.Module):
    def __init__(self, version=1.0, class_num=1000):
        super(NetCrop, self).__init__()
        self.class_num = class_num
        if version == 1.0:
            self.model = NU_squeezeNet1_0()
        if version == 1.1:
            self.model = NU_squeezeNet1_1()
        if version == 1.2:
            self.model = NU_squeezeNet1_2()
        self.crop()

    def crop(self):
        for key, module in self.model._modules.items():
            #  删除指定网络层
            if key == 'classifier':
                del self.model._modules[key]
        # print(self.model)

    def forward(self, x):
        for key, module in self.model._modules.items():
            x = module(x)
        return x


class NU_squeezeNet_SSD(nn.Module):
    def __init__(self, classes_number=1):
        super(NU_squeezeNet_SSD, self).__init__()
        self.classes_number = classes_number
        self.features = NetCrop().model.features
        extra_features = list()
        extra_features.append(nn.Conv2d(512, 256, kernel_size=1))
        extra_features.append(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1))
        extra_features.append(nn.Conv2d(512, 128, kernel_size=1))
        extra_features.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        extra_features.append(nn.Conv2d(256, 128, kernel_size=1))
        # 在特征图层为5*5和3*3时候使用较好，分别得到3*3和1*1大小的特征图层，而没有添加额外的无用信息
        extra_features.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0))
        # extra_features.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.extra_features = nn.Sequential(*extra_features)

        self.conf_layers = []
        self.conf_layers.append(nn.Conv2d(512, 1 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(512, 3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(512, 3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(256,  3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(256, 3 * self.classes_number, 1, 1, 0))
        self.conf_model = nn.ModuleList(self.conf_layers)

        self.loc_layers = []
        self.loc_layers.append(nn.Conv2d(512, 1 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(512, 3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(512, 3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(256,  3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(256, 3 * 4, 1, 1, 0))
        self.loc_model = nn.ModuleList(self.loc_layers)

    def forward(self, x):
        base_feature_indices = [10, 12]
        extra_feature_indices = [1, 3, 5]
        # extra_feature_indices = [1, 3]
        conf, loc = list(), list()
        for i, feature in enumerate(self.features):
            # print(i, feature)  # 查看这边的i决定base_feature_indices的选择
            x = feature(x)
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
                index = extra_feature_indices.index(i) + 2
                conf_predict = self.conf_model[index](x).permute(0, 2, 3, 1).contiguous()
                loc_predict = self.loc_model[index](x).permute(0, 2, 3, 1).contiguous()
                conf.append(conf_predict.view(conf_predict.size(0), -1, self.classes_number))
                loc.append(loc_predict.view(loc_predict.size(0), -1, 4))
        conf = torch.cat(conf, dim=1)
        loc = torch.cat(loc, dim=1)
        return conf, loc


class NU_squeezeNet1_2_SSD(nn.Module):
    def __init__(self, classes_number=1):
        super(NU_squeezeNet1_2_SSD, self).__init__()
        self.classes_number = classes_number
        self.features = NetCrop(version=1.2).model.features
        extra_features = list()
        extra_features.append(nn.Conv2d(256, 128, kernel_size=1))
        extra_features.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        extra_features.append(nn.Conv2d(256, 64, kernel_size=1))
        extra_features.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        extra_features.append(nn.Conv2d(128, 64, kernel_size=1))
        # 在特征图层为5*5和3*3时候使用较好，分别得到3*3和1*1大小的特征图层，而没有添加额外的无用信息
        extra_features.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0))
        # extra_features.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.extra_features = nn.Sequential(*extra_features)

        self.conf_layers = []
        self.conf_layers.append(nn.Conv2d(64, 1 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(128, 3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(256, 3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(256,  3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(128, 3 * self.classes_number, 3, 1, 1))
        self.conf_layers.append(nn.Conv2d(128, 3 * self.classes_number, 1, 1, 0))
        self.conf_model = nn.ModuleList(self.conf_layers)

        self.loc_layers = []
        self.loc_layers.append(nn.Conv2d(64, 1 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(128, 3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(256, 3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(256,  3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(128, 3 * 4, 3, 1, 1))
        self.loc_layers.append(nn.Conv2d(128, 3 * 4, 1, 1, 0))
        self.loc_model = nn.ModuleList(self.loc_layers)

    def forward(self, x):
        base_feature_indices = [9, 11, 13]
        extra_feature_indices = [1, 3, 5]
        # extra_feature_indices = [1, 3]
        conf, loc = list(), list()
        for i, feature in enumerate(self.features):
            # print(i, feature)  # 查看这边的i决定base_feature_indices的选择
            x = feature(x)
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