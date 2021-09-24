import itertools as it
import os
from collections import Counter

import numpy as np
import numpy.random as random
import pandas as pd
import timm
import torch
from scipy.io import loadmat, savemat
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from image_encoder.cub2011 import Cub2011
from image_encoder.imagenet import ImageNet

IDENTITY = 'Image Encoder| '

class ImageEncoder():
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

        self.train_loc = cub.train_loc
        self.validate_loc = cub.validate_loc
        self.test_unseen_loc = cub.test_unseen_loc
        self.test_seen_loc = cub.test_seen_loc
        self.targets = cub.targets

        visual_features = []
        for img in tqdm(cub.img_paths, desc=f"({self.config['split']}): Extracting Visual Features"):
            img = self.loader(img)
            img = self._encode(img)
            visual_features.append(img)

        self.features = [feature.numpy() for feature in visual_features]

    def _encode_imagenet(self):
        imagenet = ImageNet(config=self.config, root='/project/data/')

        features_train = []
        features_val = []
        features = []
        features_wnid = []
        for img_path in tqdm(imagenet.img_paths, desc=f"({self.config['split']}): Extracting Visual Features"):
            img = self.loader(img_path)
            img = self._encode(img)
            features_wnid.append(imagenet.wnid_to_id[imagenet.wnid_to_id["wnid"] == img_path.split('/')[3]].wnid.values[0])
            # features_train.append(img) if img_id in imagenet.seen_id else features_val.append(img)
            # features_train.append(img) if 'train' in img_path else features_val.append(img)
            features.append(img)

        self.features = np.stack(features)
        self.features_wnid = np.stack(features_wnid)

        # self.features_train = np.stack(features_train)
        # self.features_val = np.stack(features_val)
