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

IDENTITY = '  Data Preprocess ~| '

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('config/dataloader.yaml', 'r') as d, open('config/context_encoder.yaml', 'r') as c:
    image_config = yaml.safe_load(d)
    text_config = yaml.safe_load(c)
    config = image_config.copy()
    config.update(text_config)


image_mat_path = f"data/CUB_200_2011/mat/visual/{config['image_encoder']}.mat"
text_mat_path = f"data/CUB_200_2011/mat/text/{config['text_encoder']}_{str(config['semantic_sampling'])}.mat"
data_dic_path = f"data/CUB_200_2011/mat/{config['image_encoder']}_{config['split'][:-4]}_{config['text_encoder']}_{str(config['semantic_sampling'])}.mat"

if os.path.exists(data_dic_path):
    print(f'\n{IDENTITY} {data_dic_path} already exists.')
    sys.exit()

if os.path.exists(image_mat_path):
    print(f'\n{IDENTITY} .mat exists. Loading {image_mat_path} instead of encoding images.')
    image_dic = loadmat(image_mat_path)
else:
    image_encoder = ImageEncoder(config)
    image_dic = {'features': np.stack(image_encoder.features),
                 'seen_id': image_encoder.seen_id,
                 'unseen_id': image_encoder.unseen_id,
                 'train_loc': image_encoder.train_loc,
                 'test_seen_loc': image_encoder.test_seen_loc,
                 'test_unseen_loc': image_encoder.test_unseen_loc,
                 'image_config': image_config}
    savemat(image_mat_path, image_dic)

if os.path.exists(text_mat_path):
    print(f'\n{IDENTITY} .mat exists. Loading {text_mat_path} instead of encoding text.')
    semantic_dic = loadmat(text_mat_path)
else:
    context_encoder = ContextEncoder(config, device=device)
    if config['semantic_sampling']:
        label = loadmat('data/CUB_200_2011/mat/label.mat')
        semantic_distribution = SemanticDistribution(torch.tensor(context_encoder.attributes), torch.ones(context_encoder.attributes.shape[1]))
        att_train = torch.stack([semantic_distribution.sample(num_samples=1, n_points=1, context=l[0]).reshape(1, -1) 
                                 for l in label['train_idx'] - 1]).squeeze()
        semantic_dic = {'features': context_encoder.attributes,
                        'text_config': text_config,
                        'att_train': np.array(att_train)}
    else:
        semantic_dic = {'features': context_encoder.attributes,
                        'text_config': text_config}
    savemat(text_mat_path, semantic_dic)


data = loadmat("data/data/CUB/data.mat")
data_dic = {'att_train': 0,  # np.array(semantic_dic['att_train']),
            'attribute': semantic_dic['features'],
            'seen_pro': np.squeeze(semantic_dic['features'][image_dic['seen_id'] - 1]),
            'unseen_pro': np.squeeze(semantic_dic['features'][image_dic['unseen_id'] - 1]),
            'train_fea': np.squeeze(image_dic['features'][image_dic['train_loc'] - 1]),
            'test_seen_fea': np.squeeze(image_dic['features'][image_dic['test_seen_loc'] - 1]),
            'test_unseen_fea': np.squeeze(image_dic['features'][image_dic['test_unseen_loc'] - 1])}

with open(f"{data_dic_path[:-3]}yaml", "w") as f:
    yaml.dump(config, f)
print(f'\n{IDENTITY} .mat file saved to:  {data_dic_path}')
savemat(data_dic_path, data_dic)
