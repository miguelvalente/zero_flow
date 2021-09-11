import itertools as it
import os
from collections import Counter

import numpy.random as random
import pandas as pd
import torch
from scipy.io import loadmat, savemat
from torchvision.datasets.utils import download_url
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np

IDENTITY = 'Visual Encoder| '

class Cub2011(Dataset):
    '''
    Base dataset class for encoding cub
    '''
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'

    def __init__(self, root, encoder=None, config=None, download=False):
        self.root = os.path.expanduser(root)
        self.encoder = encoder
        self.config = config

        if download:
            self._download()

        self._check_integrity()

    def __len__(self):
        return len(self.data)

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _check_integrity(self):
        self._load_metadata()

        # loop checks if files and dirs exist
        try:
            for index, row in self.data.iterrows():
                filepath = os.path.join(self.root, self.base_folder, row.filepath)
                if not os.path.isfile(filepath):
                    print(f'{IDENTITY} Filepath not found: {filepath}')
        except Exception:
            return False

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', self.config['split']),
                                       sep=' ', names=['img_id', 'split', 'seen_unseen'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        self.img_paths = [os.path.join(self.root, self.base_folder, img) for img in self.data['filepath']]  # Maybe.tolist()

        self.seen_id = self.data[self.data['seen_unseen'] == 1].target.unique()
        self.unseen_id = self.data[self.data['seen_unseen'] == 0].target.unique()

        self.train = self.data[self.data['split'] == 1].img_id.values
        self.validate = self.data[self.data['split'] == 0].img_id.values
        self.test_unseen = self.data[(self.data.split == 0) & (self.data.seen_unseen == 0)].img_id.values
        self.test_seen = self.data[(self.data.split == 0) & (self.data.seen_unseen == 1)].img_id.values
        self.targets = self.data.target.values
