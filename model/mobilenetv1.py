"""
    Mobilenet V1, implemented in PyTorch.
    Original paper: https://arxiv.org/pdf/1704.04861.pdf.
"""

import torch
import torch.nn as nn
import math

__all__ = ['MobileNet', 'mobilenet_v1']


class MobileNet(nn.Module):
    def __init__(self, w_mult, n_class=1000, ps=0.2):
        super(MobileNet, self).__init__()
        self.w_mult = w_mult
        self.ps = ps
        
        def conv_bn_relu(nin, nout, k, stride=1, width_mult=1):
            assert stride in [1, 2]
            nout = int(width_mult * nout)
            return nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=k, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(num_features=nout),
                nn.ReLU(inplace=True))
        
        def dws_conv(nin, nout, width_mult=1, pool=True):
            """Depthwise Separable Convolution block"""
            stride=2 if pool else 1
            nin = int(width_mult * nin)
            nout = int(width_mult * nout)
            return nn.Sequential(
                # depthwise conv
                nn.Conv2d(nin, nin, kernel_size=3, padding=1, stride=stride, groups=nin, bias=False),
                nn.BatchNorm2d(num_features=nin),
                nn.ReLU(inplace=True),
                # pointwise conv
                nn.Conv2d(nin, nout, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_features=nout),
                nn.ReLU(inplace=True))
        
        self.model = nn.Sequential(                                       # output dim
            conv_bn_relu(3, 32, k=3, stride=2, width_mult=self.w_mult),   # 112×112×32
            dws_conv(32, 64, width_mult=self.w_mult, pool=False),         # 112×112×64
            dws_conv(64, 128, width_mult=self.w_mult, pool=True),         # 56×56×128 
            dws_conv(128, 128, width_mult=self.w_mult, pool=False),       # 56×56×128 
            dws_conv(128, 256, width_mult=self.w_mult, pool=True),        # 28×28×256 
            dws_conv(256, 256, width_mult=self.w_mult, pool=False),       # 28×28×256 
            dws_conv(256, 512, width_mult=self.w_mult, pool=True),        # 14×14×512
            
            dws_conv(512, 512, width_mult=self.w_mult, pool=False),       # 14×14×512
            dws_conv(512, 512, width_mult=self.w_mult, pool=False),       # 14×14×512
            dws_conv(512, 512, width_mult=self.w_mult, pool=False),       # 14×14×512
            dws_conv(512, 512, width_mult=self.w_mult, pool=False),       # 14×14×512
            dws_conv(512, 512, width_mult=self.w_mult, pool=False),       # 14×14×512
            
            dws_conv(512, 1024, width_mult=self.w_mult, pool=True),       # 7×7×1024
            dws_conv(1024, 1024, width_mult=self.w_mult, pool=False))     # 3×3×1024
        self.fc = nn.Sequential(
            nn.Dropout(self.ps),
            nn.Linear(1024, n_class))
        self._initialize_weights()
        
    def forward(self, x):
        x = self.model(x)
        x = x.mean(3).mean(2)
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                

def mobilenet_v1(num_classes=9):
    return MobileNet(n_class=num_classes, w_mult=1.)
     
                
                
