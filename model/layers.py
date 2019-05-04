import torch
import torch.nn as nn
import math


class Flatten(nn.Module):      
    def forward(self, inp): return input.view(inp.size(0), -1)

    
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


class SELayer(torch.nn.Module):
    """Squeeze and Excitation layer"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
