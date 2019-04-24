"""Defines the neural network, losss function and metrics"""

import torch
import torch.nn as nn
import numpy as np
import torchvision
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18']
model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}


# ================================
# Baseline
# ================================
class Net(torch.nn.Module):
    def __init__(self, K=10):
        super(Net, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1),  # 6  x 224
            torch.nn.MaxPool2d(2, 2),                         # 6  x 112
            torch.nn.Conv2d(6, 16, kernel_size=3, padding=1), # 16 x 112
            torch.nn.MaxPool2d(2, 2),                         # 16 x 56
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            Flatten(),
            torch.nn.Linear(16, 64),
            torch.nn.LeakyReLU(inplace=True),       
            torch.nn.Linear(64, 9))
               
    def forward(self, x):
        x = self.conv(x)
        return x
    

# ================================
# Supporter utils
# ================================
class Flatten(torch.nn.Module):      
    def forward(self, input): 
        return input.view(input.size(0), -1)

    
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, output_size):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(output_size)
        self.mp = nn.AdaptiveMaxPool2d(output_size)
        
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)
    
    
def custom_head(num_classes=1000, num_feat=1024, ps=0.5):
    """Head leveraged from fast.ai library. Dropout assigned in params.json.
    ps assigned to last fc layer. The layer before has ps/2. The same as in 
    fast.ai."""
    return nn.Sequential(
        Flatten(),
        nn.BatchNorm1d(num_features=num_feat),
        nn.Dropout(p=ps/2),
        nn.Linear(in_features=num_feat, out_features=num_feat // 2, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(num_features=num_feat // 2),
        nn.Dropout(p=ps),
        nn.Linear(in_features=num_feat // 2, out_features=num_classes, bias=True),
    )
    
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# class ResNet(nn.Module):

#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                   width_per_group=64, norm_layer=None):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         planes = [int(width_per_group *  2 ** i) for i in range(4)]
#         self.inplanes = planes[0]
#         self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(planes[0])
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, planes[0], layers[0],  norm_layer=norm_layer)
#         self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2,  norm_layer=norm_layer)
#         self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2,  norm_layer=norm_layer)
#         self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2,  norm_layer=norm_layer)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # TODO: custom head
#         self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, stride=1,  norm_layer=None):
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample,  norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes,  norm_layer=norm_layer))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
        
#         x = self.fc(x)

#         return x
    

# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,  norm_layer=None): 
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
# ================================
# SEResnet18
# ================================    
class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.se = SELayer(planes, 16)   # reduction=16
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
class ChannelAttentionLayer(torch.nn.Module):
    
    # https://arxiv.org/pdf/1807.06514.pdf
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
            #nn.BatchNorm2d(num_features=channel)) #
        
    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        channel_attention = self.fc(out).view(b, c, 1, 1)
        return channel_attention


class SpatialAttentionLayer(torch.nn.Module):
    
    # https://arxiv.org/pdf/1807.06514.pdf
    def __init__(self, channel, dilation=3, reduction=4):
        """Spatial attention branch.The spatial branch produces a spatial attention 
        maps emphasize or suppress features in different spatial location.
        Inputs:
        - dilation: he dilation value determines the sizeof receptive fields which 
          is helpful for the contextual information aggregation at the spatial branch
        - reduction: The reduc-tion ratio controls the capacity and overhead in both 
          attention branches
        Through the experimental validation (seeSec. 4.1), we set {d = 4, r = 16}."""
        super(SpatialAttentionLayer, self).__init__()
        self.reduce_channels = torch.nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.extract_context = nn.Sequential(
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size=3, padding=3, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=channel // reduction),
            nn.Dropout(p=0.25),
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size=3, padding=3, dilation=dilation))
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channel // reduction, 1, kernel_size=1),
            nn.BatchNorm2d(num_features=1))

    def forward(self, x):
        out = self.reduce_channels(x)
        out = self.extract_context(out)
        out = self.spatial_attention(out)
        
        return out
    
    
class CBAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.ca = ChannelAttentionLayer(planes)
        self.sa = SpatialAttentionLayer(planes)
        self.bam_sigmoid = nn.Sigmoid()
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        ca = self.ca(out)
        sa = self.sa(out)
        
        cbam = ca + sa
        cbam = self.bam_sigmoid(cbam)
        cbam = cbam * out
        
        if self.downsample is not None:
            residual = self.downsample(x)

        cbam += residual
        cbam = self.relu(cbam)

        return cbam
    
    
def se_resnet18(pretrained=False, num_classes=9):
    """We don't have weights for this model."""
    model = ResNet(block=SEBasicBlock, layers=[2, 2, 2, 2])
    model.avgpool = AdaptiveConcatPool2d(1)
    model.fc = custom_head(num_classes)
    return model


def cbam_resnet18(pretrained=False, num_classes=9):
    model = ResNet(block=CBAMBasicBlock, layers=[2, 2, 2, 2])
    model.avgpool = AdaptiveConcatPool2d(1)
    model.fc = custom_head(num_classes)
    return model


def resnet18(pretrained=False, num_classes=9, ps=0.5):
    model = ResNet(BasicBlock, [2, 2, 2, 2])   
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.avgpool = AdaptiveConcatPool2d(1)
    model.fc = custom_head(num_classes, ps=ps)
    return model

def resnet34(pretrained=False, num_classes=9):
    model = torchvision.models.resnet34(pretrained)
    model.avgpool = AdaptiveConcatPool2d(1)
    model.fc = custom_head(num_classes=num_classes, num_feat=1024)
    return model

def resnet34_fastai(pretrained=False, num_classes=9):
    model = torchvision.models.resnet34(True)
    model.avgpool = AdaptiveConcatPool2d(1)
    model.fc = custom_head(num_classes=num_classes, num_feat=512*2)
    return model

def resnet152(pretrained=False, num_classes=9):
    model = torchvision.models.resnet152(True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = custom_head(num_classes=10, num_feat=2048)
    return model


def mobilenetv2(pretrained=False, num_classes=1000):
    return model
    
    
# def loss_fn(outputs, labels):
#     return torch.nn.CrossEntropyLoss()
