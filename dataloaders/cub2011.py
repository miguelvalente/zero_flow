import itertools as it
import os
from collections import Counter

import numpy.random as random
import pandas as pd
import scipy.io
import torch
from scipy.io import savemat
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm


class Cub2011(Dataset):
    '''
    Base dataset class for loading up cub
    '''
    base_folder = '/project/data/CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, zero_shot=False, zero_test=None, which_split=None, 
                 transform=None, config=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.which_split = which_split
        self.transform = transform
        self.config = config
        self.zero_shot = zero_shot
        self.zero_test = zero_test
        self.loader = default_loader
        self.evaluation = False

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def __getitem__(self, idx):
        if self.zero_shot:
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
        else:
            img = self.visual_features[idx]
            target = self.targets[idx]

            return img, target

    def __len__(self):
        return len(self.data)
    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            print('Error loading metadata')
            return False

        # loop checks if files and dirs exist
        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False

        try:
            if self.config['load_precomputed_visual']:
                self._load_precomputed(self.config['mat_file'])
            else:
                self._encode_images()
        except Exception:
            print('Error loading visual features')
            return False

        return True

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
            if self.zero_test:
                if self.zero_test == 'seen_only':
                    self.data = self.data[self.data['is_training_img'] == 1]
                elif self.zero_test == 'unseen_only':
                    self.data = self.data[self.data['is_training_img'] == 0]
                else:
                    raise Exception:
                        print('Cannot perform zero_shot test without specifying the set of data')
            else:
                self.generation_ids = self.data['target'].unique() - 1

                self.targets = list(self.data['target'] - 1)
                self.imgs_per_class = Counter(self.targets)
                self.seen_unseen = self.data['is_training_img'].to_list()

            # self.test_gen = []
            # self.test_real = []
        else:
            if self.which_split == 'train':
                self.data = self.data[self.data['is_training_img'] == 1]
            elif self.which_split == 'val':
                self.data = self.data[self.data['is_training_img'] == 0]
            elif self.which_split == 'test':
                self.data = self.data[self.data['is_training_img'] == 2]
            elif self.which_split == 'full':
                print(f"Using entire dataset, CONFIG[save_features] = {self.config['save_visual_features']}")
            else:
                print('Split role not defined')

    def _load_precomputed(self, mat_file):
        print(f'##### Loading .mat file: {mat_file}')
        if not os.path.exists(mat_file):
            print('##### .mat does not exist. Change config[load_precomputed_visual] to False')

        raw_mat = scipy.io.loadmat(mat_file)
        img_ids = self.data['img_id'].to_numpy() - 1

        if mat_file.split('/')[-1] == 'res101.mat':
            raw_labels = raw_mat['labels'].transpose()[-1].tolist()
            raw_features = raw_mat['features'].transpose().tolist()

            # Ensures that features are ordered in ascending order according to labels
            raw_features = [v for _, v in sorted(zip(raw_labels, raw_features), key=lambda pair: pair[0])]
        else:
            raw_features = raw_mat['features'].tolist()
            raw_labels = raw_mat['labels'].tolist()

        features = torch.stack([torch.tensor(v).type(torch.float32) for v in raw_features])

        self.visual_features = features[img_ids]
        self.targets = list(self.data['target'] - 1)

    def _encode_images(self):
        save_path = f"{self.config['save_visual_features']}{self.config['image_encoder']}.mat"

        if os.path.exists(save_path):
            print("##### Model already used to extract visual features. Using _load_precomputed")
            self._load_precomputed(save_path)
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

