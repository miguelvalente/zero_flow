import os
import pickle
import sys

import numpy as np
import torch
import tqdm
import yaml
import h5py 
from scipy.io import loadmat, savemat

from distributions import SemanticDistribution
from image_encoder.image_encoder import ImageEncoder
from text_encoders.context_encoder import ContextEncoder
import pandas as pd

IDENTITY = '  Data Preprocess ~| '
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pickle_to_mat(path, config):
    dir_data = 'image_net'

    path = 'data/image_net/mat/text/ALBERT_ImageNet_trainval_classes_classes.pkl'
    splits = loadmat('data/image_net/ImageNet_splits.mat')

    seen = np.sort(splits['train_classes'].squeeze())
    unseen = np.sort(splits['val_classes'].squeeze())
    with open(path, 'rb') as f:
        data = pickle.load(f)
    corre = pd.read_csv('data/image_net/wnid_correspondance.csv', sep=' ', names=['id', 'wnid'])

    train_att = []
    val_att = []
    att = []
    wnid_att_train = []
    wnid_att_test = []
    for k, v in data.items():
        att.append(v['feats'])
        img_id = corre[corre["wnid"] == v['wnid']].id.values[0]
        train_att.append(v['feats']) if img_id in seen else val_att.append(v['feats'])
        wnid_att_train.append(v['wnid']) if img_id in seen else wnid_att_test.append(v['wnid'])

    train_att = np.stack(train_att)
    val_att = np.stack(val_att)
    att = np.stack(att)
    wnid_att_train = np.stack(wnid_att_train)
    wnid_att_test = np.stack(wnid_att_test)

    semantic_dic = {'features': att,
                    'train_att': train_att,
                    'val_att': val_att,
                    'wnid_train': wnid_att_train,
                    'wnid_test': wnid_att_test}

    text_mat_path = f"data/{dir_data}/mat/text/{config['text_encoder']}_{str(config['semantic_sampling'])}.mat"

    savemat(text_mat_path, semantic_dic)
    sys.exit()


with open('config/dataloader.yaml', 'r') as d, open('config/context_encoder.yaml', 'r') as c:
    image_config = yaml.safe_load(d)
    text_config = yaml.safe_load(c)
    config = image_config.copy()
    config.update(text_config)

if config['dataset'] == 'cub2011':
    dir_data = 'CUB_200_2011'
else:
    dir_data = 'image_net'

image_mat_path = f"data/{dir_data}/mat/visual/{config['image_encoder']}_{config['split'][:-4]}.mat"
text_mat_path = f"data/{dir_data}/mat/text/{config['text_encoder']}_{str(config['preprocess_text'])}.mat"
data_dic_path = f"data/{dir_data}/mat/{config['image_encoder']}_{config['split'][:-4]}_{config['text_encoder']}_{str(config['semantic_sampling'])}.mat"
print(f"{data_dic_path}")

# easy_split_train = loadmat("data/CUB_200_2011/cizsl/CUB2011/pfc_feat_train.mat")
# easy_split_test = loadmat("data/CUB_200_2011/cizsl/CUB2011/pfc_feat_test.mat")
# train_test_split_dir = loadmat('data/CUB_200_2011/cizsl/CUB2011/train_test_split_hard.mat')
# # pickle_to_mat('asd', config)

if os.path.exists(data_dic_path):
    print(f'\n{IDENTITY} {data_dic_path} already exists.')
    sys.exit()

if os.path.exists(image_mat_path):
    print(f'\n{IDENTITY} .mat exists. Loading {image_mat_path} instead of encoding images.')
    image_dic = loadmat(image_mat_path)
else:
    if config['dataset'] == 'cub2011':
        image_encoder = ImageEncoder(config)
        image_dic = {'features': np.stack(image_encoder.features),
                     'seen_id': image_encoder.seen_id,
                     'unseen_id': image_encoder.unseen_id,
                     'train_loc': image_encoder.train_loc,
                     'validate_loc': image_encoder.validate_loc,
                     'test_seen_loc': image_encoder.test_seen_loc,
                     'test_unseen_loc': image_encoder.test_unseen_loc,
                     'image_config': image_config}
        savemat(image_mat_path, image_dic)
    else:
        pass
        #     with h5py.File('data/image_net/ILSVRC2012_res101_feature.mat', 'r') as f:
        #         labels = np.array(f['labels'], dtype=np.int64).flatten()
        #         labels_val = np.array(f['labels_val'], dtype=np.int64).flatten()

        #         features = np.array(f['features'])
        #         features_val = np.array(f['features_val'])
        #     image_dic = {'features': features,
        #                  'features_val': features_val,
        #                  'labels': labels,
        #                  'labels_val': labels_val}

if os.path.exists(text_mat_path):
    print(f'\n{IDENTITY} .mat exists. Loading {text_mat_path} instead of encoding text.')
    semantic_dic = loadmat(text_mat_path)
else:
    context_encoder = ContextEncoder(config, device=device)
    if config['dataset'] == 'cub2011':
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
    else:
        path = 'data/image_net/mat/text/ALBERT_ImageNet_trainval_classes_classes.pkl'
        splits = loadmat('data/image_net/ImageNet_splits.mat')

        seen = np.sort(splits['train_classes'].squeeze())
        unseen = np.sort(splits['val_classes'].squeeze())
        with open(path, 'rb') as f:
            data = pickle.load(f)
        corre = context_encoder.wnid_correspondance
        corre = corre.items()
        data_list = list(corre)

        corre = pd.DataFrame(data_list)
        corre = corre.rename({0: 'wnid', 1: 'id'}, axis='columns')

        train_att = []
        val_att = []
        wnid_att_train = []
        wnid_att_test = []

        for idx, (k, v) in enumerate(data.items()):
            img_id = corre[corre["wnid"] == v['wnid']].id.values[0]
            train_att.append(context_encoder.attributes[idx]) if img_id in seen else val_att.append(context_encoder.attributes[idx])
            wnid_att_train.append(v['wnid']) if img_id in seen else wnid_att_test.append(v['wnid'])

        train_att = np.stack(train_att)
        val_att = np.stack(val_att)
        wnid_att_train = np.stack(wnid_att_train)
        wnid_att_test = np.stack(wnid_att_test)

        semantic_dic = {'features': context_encoder.attributes,
                        'train_att': train_att,
                        'val_att': val_att,
                        'wnid_train': wnid_att_train,
                        'wnid_test': wnid_att_test,
                        'text_config': text_config}
    path = 'data/image_net/mat/text/ALBERT_ImageNet_trainval_classes_classes.pkl'
    savemat(text_mat_path, semantic_dic)
    with open(f"{text_mat_path[:-3]}yaml", "w") as f:
        yaml.dump(config, f)

if config['dataset'] == "cub2011":
    # data = loadmat("data/data/CUB/data.mat")
    data_dic = {'att_train': 0,  # np.array(semantic_dic['att_train']),
                'attribute': semantic_dic['features'],
                'seen_pro': np.squeeze(semantic_dic['features'][image_dic['seen_id'] - 1]),
                'unseen_pro': np.squeeze(semantic_dic['features'][image_dic['unseen_id'] - 1]),
                'train_fea': np.squeeze(image_dic['features'][image_dic['train_loc'] - 1]),
                'val_fea': np.squeeze(image_dic['features'][image_dic['validate_loc'] - 1]),
                'test_seen_fea': np.squeeze(image_dic['features'][image_dic['test_seen_loc'] - 1]),
                'test_unseen_fea': np.squeeze(image_dic['features'][image_dic['test_unseen_loc'] - 1])}
else:
    sys.exit()
    # data_dic = {'att_train': 0,  # np.array(semantic_dic['att_train']),
    #             'attribute': semantic_dic['features'],
    #             'seen_pro': np.squeeze(semantic_dic['features'][image_dic['seen_id'] - 1]),
    #             'unseen_pro': np.squeeze(semantic_dic['features'][image_dic['unseen_id'] - 1]),
    #             'train_fea': np.squeeze(image_dic['features'][image_dic['train_loc'] - 1]),
    #             'test_seen_fea': np.squeeze(image_dic['features'][image_dic['test_seen_loc'] - 1]),
    #             'test_unseen_fea': np.squeeze(image_dic['features'][image_dic['test_unseen_loc'] - 1])}

with open(f"{data_dic_path[:-3]}yaml", "w") as f:
    yaml.dump(config, f)
print(f'\n{IDENTITY} .mat file saved to:  {data_dic_path}')
savemat(data_dic_path, data_dic)
