import os

from text_encoders.text_encoder import AlbertEncoder
import numpy as np
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import wandb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import yaml
from nets import Classifier
from text_encoders.context_encoder import ContextEncoder
from dataloaders.cub2011 import Cub2011_Pre
from convert import VisualExtractor
import scipy.io

cub = Cub2011_Pre(root='/project/data/', test=config['test'], split=config['split'], config=config, transform=transforms_cub)
