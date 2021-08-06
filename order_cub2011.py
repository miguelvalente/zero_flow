from scipy.io import loadmat, savemat
import numpy as np


def order_res101mat(path, savepath='data/xlsa17/data/CUB/res101_ordered.mat'):
    res_raw = loadmat(path)
    labels = res_raw['labels']
    features = res_raw['features'].transpose()

    features_ordered = np.stack([f for _, f in sorted(zip(labels, features), key=lambda pair: pair[0])])
    savemat(savepath, {'features': features_ordered})


def order_att_splits(path, savepath='data/CUB_200_2011/mat/text/att_splits_ordered.mat'):
    att_raw = loadmat(path)
    att = att_raw['original_att'].transpose()
    with open('data/xlsa17/data/CUB/allclasses.txt') as f:
        all_classes = f.readlines()

    labels = [int(line[:3]) for line in all_classes]
    att_ordered = np.stack([f for _, f in sorted(zip(labels, att), key=lambda pair: pair[0])])

    savemat(savepath, {'att': att_ordered})


# order_res101mat('data/xlsa17/data/CUB/res101.mat')
order_att_splits('data/xlsa17/data/CUB/att_splits.mat')
