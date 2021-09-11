import os

import numpy as np
import timm
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import yaml
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from visual_encoder.visual_encoder import VisualEncoder

from text_encoders.context_encoder import ContextEncoder
from scipy.io import loadmat, savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('config/dataloader.yaml', 'r') as d, open('config/context_encoder.yaml', 'r') as c:
    config = yaml.safe_load(d)
    config.update(yaml.safe_load(c))

data = loadmat("data/data/CUB/data.mat")
label = loadmat("data/data/CUB/label.mat")

encoder = VisualEncoder(config)
encoder.seen_id
encoder.unseen_id
encoder.train
encoder.validate
encoder.test_unseen
encoder.test_seen
encoder.targets

features = torch.arange((11788 * 2048)).reshape(11788, -1)
context_encoder = ContextEncoder(config, device=device)
contexts = context_encoder.contexts.to(device)

# mdic = {'features': features, 'labels': self.targets}
# savemat(save_path, mdic)