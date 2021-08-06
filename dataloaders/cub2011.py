import itertools as it
import os
from collections import Counter

import numpy.random as random
import pandas as pd
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm

IDENTITY = 'CUB Dataloader| '

class Cub2011(Dataset):
    '''
    Base dataset class for loading up cub
    '''
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, zero_shot=False, which_split=None,
                 transform=None, config=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.which_split = which_split
        self.transform = transform
        self.config = config
        self.zero_shot = zero_shot
        self.loader = default_loader
        self.evaluation = False

        if download:
            self._download()

        self._check_integrity()

    def __getitem__(self, idx):
        if self.zero_shot:
            if self.evaluation:
                img = self.visual_features[idx]
                target = self.targets[idx]
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
        else:
            img = self.visual_features[idx]
            target = self.targets[idx]

        seen_or_unseen = self.seen_unseen[idx]
        return img, target, seen_or_unseen

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            print(f'{IDENTITY} Error loading metadata')

        # loop checks if files and dirs exist
        try:
            for index, row in self.data.iterrows():
                filepath = os.path.join(self.root, self.base_folder, row.filepath)
                if not os.path.isfile(filepath):
                    print(f'{IDENTITY} Filepath not found: {filepath}')
        except Exception:
            return False

        try:
            if self.config['load_precomputed_visual']:
                self._load_features(self.config['mat_file_visual'])
            else:
                self._encode_images()
        except Exception:
            print(f'{IDENTITY} Error loading visual features')
            return False

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', self.config['split']),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        self.seen_id = self.data[self.data['is_training_img'] == 1].target.unique() - 1
        self.unseen_id = self.data[self.data['is_training_img'] == 0].target.unique() - 1
        self.test_id = self.data[self.data['is_training_img'] == 2].target.unique() - 1

        if self.zero_shot:
            if self.config['zero_test'] == 'seen':
                self.data = self.data[self.data['is_training_img'] == 1]
            elif self.config['zero_test'] == 'zero':
                self.data = self.data[self.data['is_training_img'] == 0]
            elif self.config['zero_test'] == 'seen_zero':
                self.data = self.data[(self.data.is_training_img == 0) | (self.data.is_training_img == 1)]
            elif self.config['zero_test'] == 'all':
                print(f'{IDENTITY} Using all data  points')
            else:
                print(f'{IDENTITY} Cannot perform zero_shot test without specifying the set of data')
                raise Exception

            self.generation_ids = self.data['target'].unique() - 1

            self.targets = list(self.data['target'] - 1)
            self.imgs_per_class = Counter(self.targets)

            self.test_gen = []
            self.test_real = []
        else:
            if self.which_split == 'train':
                self.data = self.data[self.data['is_training_img'] == 1]
            elif self.which_split == 'val':
                self.data = self.data[self.data['is_training_img'] == 0]
            elif self.which_split == 'test':
                self.data = self.data[self.data['is_training_img'] == 2]
            elif self.which_split == 'full':
                print(f"{IDENTITY} ({self.which_split}) Using entire dataset, CONFIG[save_features] = {self.config['save_visual_features']}")
            else:
                print('Split role not defined')

        self.seen_unseen = self.data['is_training_img'].to_list()

    def _load_features(self, mat_file):
        print(f'{IDENTITY} Loading .mat file: {mat_file}')
        if not os.path.exists(mat_file):
            print(f'{IDENTITY} .mat does not exist. Loading not possible using _encode_images() instead.')
            self._encode_images()
        else:
            raw_mat = loadmat(mat_file)
            raw_features = raw_mat['features']

            features = torch.from_numpy(raw_features).type(torch.float32)

            if self.config['minmax_rescale']:
                features = 1 * ((features - features.min(axis=0).values) / (features.max(axis=0).values - features.min(axis=0).values))

            img_ids = self.data['img_id'].to_numpy() - 1
            self.visual_features = features[img_ids]
            self.targets = list(self.data['target'] - 1)

    def _encode_images(self):
        save_path = f"{self.config['save_visual_features']}{self.config['image_encoder']}.mat"

        if os.path.exists(save_path):
            print(f"{IDENTITY} Model already used to encode images. Using _load_features() instead")
            self._load_features(save_path)
        else:
            img_paths = [os.path.join(self.root, self.base_folder, img) for img in self.data['filepath']]  # Maybe.tolist()

            self.visual_features = []

            for img in tqdm(img_paths, desc=f"({self.which_split}): Extracting Visual Features"):
                img = self.loader(img)
                if self.transform is not None:
                    img = self.transform(img)
                self.visual_features.append(img)

            self.targets = list(self.data['target'] - 1)

            if self.config['save_visual_features']:
                features = [feature.numpy() for feature in self.visual_features]
                mdic = {'features': features, 'labels': self.targets}
                savemat(save_path, mdic)

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

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
