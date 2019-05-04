import torch
import torch.nn as nn
import math
import model.mobilenetv2 as mobilenetv2


__all__ = ['mobilenet_v2']

              
def mobilenet_v2(num_classes=9):
    return mobilenetv2.MobileNetV2(n_class=num_classes, input_size=224, width_mult=1.)