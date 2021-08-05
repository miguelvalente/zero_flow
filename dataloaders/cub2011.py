import itertools as it
import os
from collections import Counter

import numpy.random as random
import pandas as pd
import scipy.io
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm


class Cub2011_Base(Dataset):
    '''
    Base dataset class for loading up cub
    '''
    base_folder = '/project/data/CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, which_split=None, test=False, split='easy_split.txt', 
                 transform=None, config=None, loader=default_loader, download=False):

        self.root = os.path.expanduser(root)
        self.which_split = which_split
        self.test = test
        self.split = split
        self.transform = transform
        self.config = config
        self.loader = default_loader
        self.evaluation = False

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

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
        return len(self.data)


class Cub2011(Cub2011_Base):
    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', self.split),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        self.seen_id = self.data[self.data.is_training_img == 1].target.unique() - 1
        self.unseen_id = self.data[self.data.is_training_img == 0].target.unique() - 1

        if self.which_split == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        elif self.which_split == 'val':
            self.data = self.data[self.data.is_training_img == 0]
        elif self.which_split == 'test':
            self.data = self.data[self.data.is_training_img == 2]
            self.test_id = self.data[self.data.is_training_img == 2].target.unique() - 1
        else:
            print('Split role not defined')

        img_paths = [os.path.join(self.root, self.base_folder, img) for img in list(self.data.filepath)]
        self.visual_features = []
        for img in tqdm(img_paths, desc=f"({self.which_split}): Extracting Visual Features"):
            img = self.loader(img)
            if self.transform is not None:
                img = self.transform(img)
            self.visual_features.append(img)
        # self.visual_features = torch.ones((10000, 2048))
        self.targets = list(self.data.target)

    def __getitem__(self, idx):
        img = self.visual_features[idx]
        target = self.targets[idx] - 1

        return img, target

class Cub2011Zero(Cub2011_Base):
    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', self.split),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        self.seen_id = self.data[self.data.is_training_img == 1].target.unique() - 1
        self.unseen_id = self.data[self.data.is_training_img == 0].target.unique() - 1
        self.test_id = self.data[self.data.is_training_img == 2].target.unique() - 1

        if not self.test:
            self.data = self.data[(self.data.is_training_img == 0) | (self.data.is_training_img == 1)]

        # Contexts to generate
        self.generation_ids = self.data.target.unique() - 1

        self.targets = list(self.data.target - 1)
        self.imgs_per_class = Counter(self.targets)
        self.seen_unseen = list(self.data.is_training_img)

        img_paths = [os.path.join(self.root, self.base_folder, img) for img in list(self.data.filepath)]
        self.visual_features = []
        for img in tqdm(img_paths, desc="Extracting Visual Features"):
            img = self.loader(img)
            if self.transform is not None:
                img = self.transform(img)
            self.visual_features.append(img)

        # self.visual_features = torch.ones((len(self.data), 2048))

        self.test_gen = []
        self.test_real = []

    def __getitem__(self, idx):
        if self.evaluation:
            img = self.visual_features[idx]
            target = self.targets[idx]
            seen_or_unseen = self.seen_unseen[idx]

            return img, target, seen_or_unseen  # unseen = 0 / seen = 1
        else:
            if random.choice(self.config['randomness']) == 0:
                target = self.targets[idx]
                if self.seen_unseen[idx] == 1:
                    self.test_real.append(target)
                    img = self.visual_features[idx]
                else:
                    self.test_gen.append(target)
                    img = self.generated_features[idx]
            else:
                img = self.generated_features[idx]
                target = self.targets[idx]
                self.test_gen.append(target)

            seen_or_unseen = self.seen_unseen[idx]
            return img, target, seen_or_unseen

    def insert_generated_features(self, generated_features, labels):
        """Function used to insert generated features for Generative Zero Shot Learning

        Parameters:
        generated_features (tensor array): Features generated for each class
        labels (int): orders labels based on new features
        """
        self.generated_features = (generated_features).reshape(-1, generated_features.shape[-1])
        self.targets = labels

    def eval(self):
        '''Function to set self.evaluation True and False to change __getitem__() return'''
        self.evaluation = False if self.evaluation else True


class Cub2011_Pre(Cub2011_Base):
    def _load_metadata(self):

        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', self.split),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        self.seen_id = self.data[self.data.is_training_img == 1].target.unique() - 1
        self.unseen_id = self.data[self.data.is_training_img == 0].target.unique() - 1

        if self.which_split == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        elif self.which_split == 'val':
            self.data = self.data[self.data.is_training_img == 0]
            self.val_id = self.data[self.data.is_training_img == 0].target.unique() - 1
        elif self.which_split == 'test':
            self.data = self.data[self.data.is_training_img == 2]
            self.test_id = self.data[self.data.is_training_img == 2].target.unique() - 1
        else:
            print('Split role not defined')

        ids = self.data['img_id'].to_numpy() - 1

        if self.config['visual_order']:
            raw_res = scipy.io.loadmat('data/xlsa17/data/CUB/res101.mat')
            raw_res = scipy.io.loadmat('data/CUB_200_2011/mat/visual/res101_ordered.mat')
            features = raw_res['features']
            self.visual_features = torch.from_numpy(features).type(torch.float32)[ids]

            # labels = raw_res['labels']
            # features = raw_res['features'].transpose()
            # features_ordered = torch.stack([torch.from_numpy(f).type(torch.float32) for _, f in sorted(zip(labels, features), key=lambda pair: pair[0])])
            # self.visual_features = features_ordered[ids]
        else:
            raw_res = scipy.io.loadmat('data/xlsa17/data/CUB/res101.mat')
            self.visual_features = torch.from_numpy(raw_res['features'].transpose()[ids]).type(torch.float32)
        self.targets = self.data['target'].to_list()

    def __getitem__(self, idx):
        img = self.visual_features[idx]
        target = self.targets[idx] - 1

        return img, target

class Cub2011Zero_Pre(Cub2011_Base):
    def _load_metadata(self):
        raw_res = scipy.io.loadmat('data/xlsa17/data/CUB/res101.mat')

        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', self.split),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        self.seen_id = self.data[self.data.is_training_img == 1].target.unique() - 1
        self.unseen_id = self.data[self.data.is_training_img == 0].target.unique() - 1
        self.test_id = self.data[self.data.is_training_img == 2].target.unique() - 1

        if not self.test:
            self.data = self.data[(self.data.is_training_img == 0) | (self.data.is_training_img == 1)]

        # Contexts to generate
        self.generation_ids = self.data.target.unique() - 1

        self.targets = list(self.data.target - 1)
        self.imgs_per_class = Counter(self.targets)
        self.seen_unseen = list(self.data.is_training_img)

        ids = self.data['img_id'].to_numpy() - 1

        self.visual_features = torch.from_numpy(raw_res['features'].transpose()[ids]).type(torch.float32)
        self.targets = self.data['target'].to_list()

        self.test_gen = []
        self.test_real = []

    def __getitem__(self, idx):
        if self.evaluation:
            img = self.visual_features[idx]
            target = self.targets[idx]
            seen_or_unseen = self.seen_unseen[idx]

            return img, target, seen_or_unseen  # unseen = 0 / seen = 1
        else:
            if random.choice(self.config['randomness']) == 0:
                target = self.targets[idx]
                if self.seen_unseen[idx] == 1:
                    self.test_real.append(target)
                    img = self.visual_features[idx]
                else:
                    self.test_gen.append(target)
                    img = self.generated_features[idx]
            else:
                img = self.generated_features[idx]
                target = self.targets[idx]
                self.test_gen.append(target)

            seen_or_unseen = self.seen_unseen[idx]
            return img, target, seen_or_unseen

    def insert_generated_features(self, generated_features, labels):
        """Function used to insert generated features for Generative Zero Shot Learning

        Parameters:
        generated_features (tensor array): Features generated for each class
        labels (int): orders labels based on new features
        """
        self.generated_features = (generated_features).reshape(-1, generated_features.shape[-1])
        self.targets = labels

    def eval(self):
        '''Function to set self.evaluation True and False to change __getitem__() return'''
        self.evaluation = False if self.evaluation else True

