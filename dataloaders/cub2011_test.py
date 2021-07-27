import os
import scipy.io

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

    def __init__(self, root, test=False, split='zsl_split_20.txt', transform=None, config=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.test = test
        self.split = split
        self.transform = transform
        self.config = config
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

        self.data = self.data[self.data.is_training_img == 0]
        # self.data = self.data[self.data.is_training_img == 1]

        # Contexts to generate
        self.generation_ids = self.data.target.unique() - 1

        self.targets = list(self.data.target - 1)
        self.imgs_per_class = Counter(self.targets)
        self.seen_unseen = list(self.data.is_training_img)

        raw_res = scipy.io.loadmat('data/xlsa17/data/CUB/res101.mat')

        ids = self.data['img_id'].to_numpy() - 1

        raw_visual = raw_res['features'].transpose().tolist()
        raw_labels = raw_res['labels'].transpose()[-1].tolist()

        visuals = [v for _, v in sorted(zip(raw_labels, raw_visual), key=lambda pair: pair[0])]
        visuals = torch.stack([torch.tensor(v).type(torch.float32) for v in visuals])

        self.visual_features = visuals[ids]  
        self.targets = self.data['target'].to_list()


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

            return img, target, seen_or_unseen  # unseen = 0 / seen = 1
        else:
            img = self.generated_features[idx]
            target = self.targets[idx]
            seen_or_unseen = self.seen_unseen[idx]

            return img, target, seen_or_unseen

    def insert_generated_features(self, generated_features, labels):
        """Function used to insert generated features for Generative Zero Shot Learning
           Also serves to set zero_shot_mode to True. This is used to calculate the lenght of the data with new unseeen features

        Parameters:
        generated_features (tensor array): Features generated for each class
        number_samples (int): number of generated samples per class
        """
        self.generated_features = (generated_features).reshape(-1, generated_features.shape[-1])
        self.targets = labels
        self.features_inserted = True

    def eval(self):
        '''Function to set self.evaluation True and False to change __getitem__() return'''
        self.evaluation = False if self.evaluation else True
