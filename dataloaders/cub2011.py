import os

from tqdm import tqdm
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import itertools


class Cub2011(Dataset):
    base_folder = '/project/data/CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, which_split, split='easy_split.txt', transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.which_split = which_split
        self.split = split
        self.transform = transform
        self.loader = default_loader
        self.zero_shot_mode = False
        self.evaluation = False

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

        self.unseen_id = self.data[self.data.is_training_img == 0].target.unique() - 1
        self.seen_id = self.data[self.data.is_training_img == 1].target.unique() - 1

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

    def __getitem__(self, idx):
        img = self.visual_features[idx]
        target = self.targets[idx] - 1

        return img, target