import itertools as it
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
from collections import Counter

import numpy.random as random
import pandas as pd
import torch
from scipy.io import loadmat, savemat
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import numpy as np
from visual_encoder.cub2011 import Cub2011

IDENTITY = 'Visual Encoder| '

class VisualEncoder():
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.loader = default_loader

        self._model_init()

        print(f"\n Visual Encoder Summary: \n Dataset: {self.config['dataset']} \
                                           \n Model: {self.config['image_encoder']} \n Transforms: {self.transform}")

        if self.config['dataset'] == 'imagenet':
            self._encode_imagenet()
        elif self.config['dataset'] == 'cub2011':
            self._encode_cub2011()
        else:
            print(f"{IDENTITY} Dataset not found")
            raise Exception

    def _encode(self, sample):
        tensor = self.transform(sample).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        return out.squeeze()

    def _model_init(self):
        model = self.config['image_encoder']
        self.model = timm.create_model(model, pretrained=True, num_classes=0)
        self.model.eval()

        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        self.transform.transforms = self.transform.transforms[:-1]

    def _encode_cub2011(self):
        cub = Cub2011(config=self.config, root='/project/data/')

        self.seen_id = cub.seen_id
        self.unseen_id = cub.unseen_id

        self.train = cub.train
        self.validate = cub.validate
        self.test_unseen = cub.test_unseen
        self.test_seen = cub.test_seen
        self.targets = cub.targets

        visual_features = []
        for img in tqdm(cub.img_paths[:10], desc=f"({self.config['split']}): Extracting Visual Features"):
            img = self.loader(img)
            img = self._encode(img)
            visual_features.append(img)

        self.features = [feature.numpy() for feature in visual_features]
