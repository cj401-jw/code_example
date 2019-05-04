import torch
import torch.nn as nn
import numpy as np
import torchvision
import math
import model.layers as layers

def loss_fn(outputs, labels):
    return torch.nn.CrossEntropyLoss()
