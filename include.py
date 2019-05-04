import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn

import operator

from pathlib import Path  # replace with os.scandir
from shutil import copyfile

import glob
import tqdm
import random
import shutil

import argparse
import logging
import os

import json
import pickle

# from fastai import *
# from fastai.vision import *

# images
import cv2
import torchvision
import torchvision.utils
import torchvision.transforms.functional as T

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style, colors
plt.ion()  # make plot interactive

import PIL

# augmentation
from imgaug import augmenters as iaa
import imgaug as ia

device = 'cuda' if torch.cuda.is_available() else 'cpu'
imagenet_stats = ([0.485, 0.456, 0.406], 
                  [0.229, 0.224, 0.225])

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True


