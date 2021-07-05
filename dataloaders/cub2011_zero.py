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

    def __init__(self, root, split='zsl_split_20.txt', transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
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

        self.data_unseen = self.data[self.data.is_training_img == 0]
        self.data = self.data[self.data.is_training_img == 1]
        self.seen_len = len(self.data)

        self.data = pd.concat([self.data, self.data_unseen])
        self.targets = list(self.data.target)
        self.seen_unseen = list(self.data.is_training_img)

        img_paths = [os.path.join(self.root, self.base_folder, img) for img in list(self.data.filepath)]
        self.visual_features = []
        for img in tqdm(img_paths, desc="Extracting Visual Features"):
            img = self.loader(img)
            if self.transform is not None:
                img = self.transform(img)
            self.visual_features.append(img)

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
        if self.evaluation:
            img = self.visual_features[idx]
            target = self.targets[idx] - 1
            seen_or_unseen = self.seen_unseen[idx]
            # sample = self.data.iloc[idx]
            # path = os.path.join(self.root, self.base_folder, sample.filepath)
            # target = sample.target - 1  # Targets start at 1 by default, so shift to 0
            # img = self.loader(path)
            # seen_or_unseen = sample.is_training_img
            # if self.transform is not None:
            #     img = self.transform(img)

            return img, target, seen_or_unseen  # unseen = 1 / seen = 0
        else:
            if idx >= self.seen_len:  # This is only true when zero_shot_mode = True
                idx = idx - self.seen_len
                target = self.data_generated_features_targets[idx] - 1  # Targets start at 1 by default, so shift to 0
                img = self.data_generated_features[idx]
            else:
                img = self.visual_features[idx]
                target = self.targets[idx] - 1
                # sample = self.data.iloc[idx]
                # path = os.path.join(self.root, self.base_folder, sample.filepath)
                # target = sample.target - 1  # Targets start at 1 by default, so shift to 0
                # img = self.loader(path)
                # if self.transform is not None:
                #     img = self.transform(img)

            return img, target

    def insert_generated_features(self, generated_unseen_features, number_samples):
        """Function used to insert unseen generated features for Generatice Zero Shot Learning
           Also serves to set zero_shot_mode to True. This is used to calculate the lenght of the data with new unseeen features

        Parameters:
        generated_unseen_features (tensor array): Features generated for each unseen class
        number_samples (int): number of generated samples per unseen class

        """
        self.data_generated_features = (generated_unseen_features).reshape(-1, generated_unseen_features.shape[-1])
        target_ids = self.data_unseen.target.unique()
        self.data_generated_features_targets = list(itertools.chain.from_iterable(itertools.repeat(target_id, 60) for target_id in target_ids))

        if not self.zero_shot_mode:
            self.zero_shot_mode = True

    def eval(self):
        '''Function to set self.evaluation True and False to change __getitem__() return'''
        self.evaluation = False if self.evaluation else True
        # if self.evaluation:
        #     self.evaluation = False
        # else:
        #     self.evaluation = True
