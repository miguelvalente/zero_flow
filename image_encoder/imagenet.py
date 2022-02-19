import itertools as it
import os

from scipy.io import loadmat
from torch.utils.data import Dataset
import numpy as np

IDENTITY = 'ImageNet Encoder| '

class ImageNet(Dataset):
    def __init__(self, root, encoder=None, config=None, download=False):
        self.root = os.path.expanduser(root)
        self.encoder = encoder
        self.config = config

        self._load_metadata()

    def __len__(self):
        return len(self.data)

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
        # self.wnid_to_id = pd.read_csv('data/image_net/wnid_correspondance.csv', sep=' ', names=['id', 'wnid'])
        self.splits = loadmat('data/image_net/ImageNet_splits.mat')
        self.seen_id = np.sort(self.splits['train_classes'].squeeze())
        self.unseen_id = np.sort(self.splits['val_classes'].squeeze())

        img_paths_train = []
        img_paths_val = []
        for (dirpath, dirnames, filenames) in os.walk('data/image_net/train'):
            img_paths_train.append([os.path.join(dirpath, filename) for filename in filenames])
        for (dirpath, dirnames, filenames) in os.walk('data/image_net/val'):
            img_paths_val.append([os.path.join(dirpath, filename) for filename in filenames])

        img_paths_train = list(it.chain(*img_paths_train))
        img_paths_val = list(it.chain(*img_paths_val))
        self.img_paths = img_paths_val + img_paths_train
