import os
import sys

import timm
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import yaml
from PIL import Image
from scipy.io import loadmat, savemat
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from distributions import DoubleDistribution, SemanticDistribution
from image_encoder.image_encoder import ImageEncoder
from text_encoders.context_encoder import ContextEncoder

IDENTITY = 'Data Preprocess | '

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('config/dataloader.yaml', 'r') as d, open('config/context_encoder.yaml', 'r') as c:
    image_config = yaml.safe_load(d)
    text_config = yaml.safe_load(c)
    config = yaml.safe_load(d)
    config = config.update(yaml.safe_load(c))

image_mat_path = f"data/CUB_200_2011/mat/visual/{config['image_encoder']}"
text_mat_path = f"data/CUB_200_2011/mat/text/{config['text_encoder']}"
data_dic_path = f"data/CUB_200_2011/mat/{config['image_encoder']}_{config['text_encoder']}"

if os.path.exists(data_dic_path):
    print(f'{IDENTITY} {data_dic_path} already exists.')
    sys.exit()

if os.path.exists(image_mat_path):
    print(f'{IDENTITY} .mat exists. Loading {image_mat_path} instead of encoding images.')
    loadmat(image_mat_path)
else:
    image_encoder = ImageEncoder(config)
    image_dic = {'features': image_encoder.features,
                 'seen_id': image_encoder.seen_id,
                 'unseen_id': image_encoder.unseen_id,
                 'train_loc': image_encoder.train_loc,
                 'test_seen_loc': image_encoder.test_seen_loc,
                 'test_unseen_loc': image_encoder.test_unseen_loc,
                 'image_config': image_config}
    savemat(f"data/CUB_200_2011/mat/visual/{config['image_encoder']}", image_dic)

if os.path.exists(text_mat_path):
    print(f'{IDENTITY} .mat exists. Loading {text_mat_path} instead of encoding text.')
    semantic_dic = loadmat(text_mat_path)
else:
    context_encoder = ContextEncoder(config, device=device)
    semantic_dic = {'attribute': context_encoder.attributes,
                    'text_config': text_config}
    savemat(f"data/CUB_200_2011/mat/text/{config['text_encoder']}", semantic_dic)

data_dic = {'att_train': 0,
            'attribute': semantic_dic['attributes'],
            'seen_pro': semantic_dic['attributes'][image_dic['seen_id'] - 1],
            'unseen_pro': semantic_dic['attributes'][image_dic['unseen_id'] - 1],
            'train_fea': image_dic['features'][image_dic['train_loc'] - 1],
            'test_seen_fea': image_dic['features'][image_dic['test_seen_loc'] - 1],
            'test_unseen_fea': image_dic['features'][image_dic['test_unseen_loc'] - 1],
            'processing_config': config}

savemat(f"data/CUB_200_2011/mat/{config['image_encoder']}_{config['text_encoder']}", data_dic)
