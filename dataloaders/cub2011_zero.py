import os

from collections import Counter
import numpy.random as random
from tqdm import tqdm
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import itertools as it


class Cub2011(Dataset):
    base_folder = '/project/data/CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, test=False, split='zsl_split_20.txt', transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.test = test
        self.split = split
        self.transform = transform
        self.loader = default_loader
        self.evaluation = False
        self.features_inserted = False

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', self.split),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        self.unseen_ids = list(self.data[self.data.is_training_img == 0].target.unique() - 1)
        self.seen_ids = list(self.data[self.data.is_training_img == 1].target.unique() - 1)
        self.test_ids = list(self.data[self.data.is_training_img == 2].target.unique() - 1)

        if not self.test:
            self.data = self.data[(self.data.is_training_img == 0) | (self.data.is_training_img == 1)]

        # Contexts to generate
        self.generation_ids = self.data.target.unique() - 1

        self.targets = list(self.data.target - 1)
        self.imgs_per_class = Counter(self.targets)
        self.seen_unseen = list(self.data.is_training_img)

        # img_paths = [os.path.join(self.root, self.base_folder, img) for img in list(self.data.filepath)]
        # self.visual_features = []
        # for img in tqdm(img_paths, desc="Extracting Visual Features"):
        #     img = self.loader(img)
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     self.visual_features.append(img)

        self.visual_features = torch.ones((len(self.data), 2048))

        self.test_gen = []
        self.test_real = []

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.visual_features)

    def __getitem__(self, idx):
        if self.evaluation:
            img = self.visual_features[idx]
            target = self.targets[idx]
            seen_or_unseen = self.seen_unseen[idx]

            return img, target, seen_or_unseen  # unseen = 1 / seen = 0
        else:
            if random.choice(3) != 0:
                target = self.targets[idx]
                if self.seen_unseen[idx] == 0:
                    self.test_real.append(target)
                    img = self.visual_features[idx]
                else:
                    self.test_gen.append(target)
                    img = self.generated_features[idx]
            else:
                img = self.generated_features[idx]
                target = self.targets[idx]

            return img, target

    def insert_generated_features(self, generated_features, labels):
        """Function used to insert unseen generated features for Generatice Zero Shot Learning
           Also serves to set zero_shot_mode to True. This is used to calculate the lenght of the data with new unseeen features

        Parameters:
        generated_unseen_features (tensor array): Features generated for each unseen class
        number_samples (int): number of generated samples per unseen class

        """
        self.generated_features = (generated_features).reshape(-1, generated_features.shape[-1])
        self.targets = labels
        self.features_inserted = True

    def eval(self):
        '''Function to set self.evaluation True and False to change __getitem__() return'''
        self.evaluation = False if self.evaluation else True
