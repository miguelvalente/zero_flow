import numpy as np
import pandas as pd
import h5py
import pickle
from scipy.io import loadmat
from sklearn import preprocessing
import torch

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label

def select_indices(labels, classes):
    assert labels.ndim == 1
    if isinstance(classes, set):
        classes = np.array(list(classes))
    mask = np.isin(labels, test_elements=classes)
    indices = np.where(mask)[0].astype(np.int64)
    return indices

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.dataset in ['FLO', 'cub2011']:
            if ('easy' in opt.split or 'hard' in opt.split) and 'benchmark' in opt.split:
                self.read_zsl_benchmark(opt)
            elif ('easy' in opt.split or 'hard' in opt.split) and 'benchmark' not in opt.split:
                self.read_zsl(opt)
            else:
                self.read_gzsl(opt)
        else:
            self.read_imagenet(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute[1].shape[0] if isinstance(self.attribute, dict) else self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        # for i in range(self.seenclasses.shape[0]):
        #     self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

    def read_zsl(self, opt):
        txt_feat_path = 'data/CUB_200_2011/cizsl/CUB_Porter_7551D_TFIDF_new.mat'
        if 'easy' in opt.split:
            train_test_split_dir = 'data/CUB_200_2011/cizsl/train_test_split_easy.mat'
            pfc_label_path_train = 'data/CUB_200_2011/cizsl/labels_train.pkl'
            pfc_label_path_test = 'data/CUB_200_2011/cizsl/labels_test.pkl'
            pfc_feat_path_train = 'data/CUB_200_2011/cizsl/pfc_feat_train.mat'  # pfc_feat (8855, 3584)
            pfc_feat_path_test = 'data/CUB_200_2011/cizsl/pfc_feat_test.mat'  # pfc_feat (2933, 3584)

            train_cls_num = 150
            test_cls_num = 50

        elif 'hard' in opt.split:
            train_test_split_dir = 'data/CUB_200_2011/cizsl/train_test_split_hard.mat'
            pfc_label_path_train = 'data/CUB_200_2011/cizsl/labels_train_hard.pkl'
            pfc_label_path_test = 'data/CUB_200_2011/cizsl/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/CUB_200_2011/cizsl/pfc_feat_train_hard.mat'  # pfc_feat (9410, 3584)
            pfc_feat_path_test = 'data/CUB_200_2011/cizsl/pfc_feat_test_hard.mat'  # pfc_feat (2378, 3584)

            train_cls_num = 160
            test_cls_num = 40

        matcontent = loadmat(opt.data_dir)

        train_att = matcontent['att_train']
        attribute = matcontent['attribute']
        train_fea = matcontent['train_fea']
        test_fea = matcontent['val_fea']

        self.pfc_feat_data_test = test_fea
        self.pfc_feat_data_train = train_fea

        # self.pfc_feat_data_train = loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        # self.pfc_feat_data_test = loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
        # Image Features Normalization
        # min_max_scaler = preprocessing.MinMaxScaler().fit(train_fea)
        # train_fea = min_max_scaler.transform(train_fea)
        # test_fea = min_max_scaler.transform(test_fea)
        if 'min' in opt.split:
            min_max_scaler = preprocessing.MinMaxScaler().fit(self.pfc_feat_data_train)
            self.pfc_feat_data_train = min_max_scaler.transform(self.pfc_feat_data_train)
            self.pfc_feat_data_test = min_max_scaler.transform(self.pfc_feat_data_test)
        else:
            mean = self.pfc_feat_data_train.mean()
            var = self.pfc_feat_data_train.var()
            self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var  # X_tr
            self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var  # X_te

        # calculate the corresponding centroid.
        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            self.labels_train = pickle.load(fout1, encoding="latin1")
            self.labels_test = pickle.load(fout2, encoding="latin1")

        self.train_cls_num = train_cls_num  # Y_train
        self.test_cls_num = test_cls_num  # Y_test
        self.feature_dim = self.pfc_feat_data_train.shape[1]

        self.tr_cls_centroid = np.zeros([self.train_cls_num, self.pfc_feat_data_train.shape[1]]).astype(np.float32)
        # print(self.tr_cls_centroid.shape) # e.g. (160, 3584)

        # calculate the feature mean of each class
        for i in range(self.train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)

        self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path,
                                                                           train_test_split_dir)  # Z_tr, Z_te
        self.att_dim = self.train_text_feature.shape[1]

        self.ntrain_class = self.train_cls_num
        self.ntest_class = self.test_cls_num
        self.train_feature = torch.tensor(self.pfc_feat_data_train)
        self.test_feature = self.pfc_feat_data_test
        self.train_label = torch.tensor(self.labels_train)
        self.test_label = self.labels_test
        self.train_att = self.train_text_feature
        self.test_att = self.test_text_feature
        self.ntrain = self.train_feature.shape[0]
        self.attribute = np.vstack((self.train_text_feature, self.test_text_feature))

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_label)

        self.test_seen_label = torch.tensor(self.train_label.clone().detach())
        self.test_unseen_label = torch.tensor(self.labels_test)

        self.test_unseen_feature = torch.tensor(self.pfc_feat_data_test)

    def read_zsl_benchmark(self, opt):
        txt_feat_path = 'data/CUB_200_2011/cizsl/CUB_Porter_7551D_TFIDF_new.mat'
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")
        if 'easy' in opt.split:
            train_test_split_dir = 'data/CUB_200_2011/cizsl/train_test_split_easy.mat'
            pfc_label_path_train = 'data/CUB_200_2011/cizsl/labels_train.pkl'
            pfc_label_path_test = 'data/CUB_200_2011/cizsl/labels_test.pkl'
            pfc_feat_path_train = 'data/CUB_200_2011/cizsl/pfc_feat_train.mat'  # pfc_feat (8855, 3584)
            pfc_feat_path_test = 'data/CUB_200_2011/cizsl/pfc_feat_test.mat'  # pfc_feat (2933, 3584)

            train_cls_num = 150
            test_cls_num = 50

        elif 'hard' in opt.split:
            train_test_split_dir = 'data/CUB_200_2011/cizsl/train_test_split_hard.mat'
            pfc_label_path_train = 'data/CUB_200_2011/cizsl/labels_train_hard.pkl'
            pfc_label_path_test = 'data/CUB_200_2011/cizsl/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/CUB_200_2011/cizsl/pfc_feat_train_hard.mat'  # pfc_feat (9410, 3584)
            pfc_feat_path_test = 'data/CUB_200_2011/cizsl/pfc_feat_test_hard.mat'  # pfc_feat (2378, 3584)

            train_cls_num = 160
            test_cls_num = 40

        self.pfc_feat_data_train = loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_test = loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
        # calculate the corresponding centroid.
        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            self.labels_train = pickle.load(fout1, encoding="latin1")
            self.labels_test = pickle.load(fout2, encoding="latin1")

        self.train_cls_num = train_cls_num  # Y_train
        self.test_cls_num = test_cls_num  # Y_test
        self.feature_dim = self.pfc_feat_data_train.shape[1]

        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()
        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var  # X_tr
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var  # X_te

        self.tr_cls_centroid = np.zeros([self.train_cls_num, self.pfc_feat_data_train.shape[1]]).astype(np.float32)
        # print(self.tr_cls_centroid.shape) # e.g. (160, 3584)

        # calculate the feature mean of each class
        for i in range(self.train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)

        self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path,
                                                                           train_test_split_dir)  # Z_tr, Z_te
        self.att_dim = self.train_text_feature.shape[1]

        self.ntrain_class = self.train_cls_num
        self.ntest_class = self.test_cls_num
        self.train_feature = torch.tensor(self.pfc_feat_data_train)
        self.test_feature = self.pfc_feat_data_test
        self.train_label = torch.tensor(self.labels_train)
        self.test_label = self.labels_test
        self.train_att = self.train_text_feature
        self.test_att = self.test_text_feature
        self.ntrain = self.train_feature.shape[0]
        self.attribute = np.vstack((self.train_text_feature, self.test_text_feature))

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_label)

        self.test_seen_label = torch.tensor(self.train_label.clone().detach())
        self.test_unseen_label = torch.tensor(self.labels_test)

        self.test_unseen_feature = torch.tensor(self.pfc_feat_data_test)

    def read_gzsl(self, opt):
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")

        matcontent = loadmat(opt.data_dir)

        train_att = matcontent['att_train']
        seen_pro = matcontent['seen_pro']
        attribute = matcontent['attribute']
        unseen_pro = matcontent['unseen_pro']
        self.attribute = torch.from_numpy(attribute).float()
        self.train_att_sampled = matcontent['att_train'].astype(np.float32)
        self.train_att = seen_pro.astype(np.float32)
        self.test_att = unseen_pro.astype(np.float32)

        train_fea = matcontent['train_fea']
        test_seen_fea = matcontent['test_seen_fea']
        test_unseen_fea = matcontent['test_unseen_fea']

        # Image Features Normalization
        min_max_scaler = preprocessing.MinMaxScaler().fit(train_fea)
        train_fea = min_max_scaler.transform(train_fea)
        test_seen_feature = min_max_scaler.transform(test_seen_fea)
        test_unseen_feature = min_max_scaler.transform(test_unseen_fea)
        # mean = train_fea.mean(axis=0)
        # train_fea -= mean
        # var = train_fea.var(axis=0)
        # train_fea /= var
        # test_seen_fea = (test_seen_fea - mean) / var
        # test_unseen_fea = (test_unseen_fea - mean) / var

        if opt.normalize_semantics:
            # Semantic Features Normalization
            min_max_scaler = preprocessing.MinMaxScaler().fit(self.train_att)
            # std_scaler = preprocessing.StandardScaler().fit(self.train_att)
            self.train_att = min_max_scaler .transform(self.train_att)
            self.test_att = min_max_scaler.transform(self.test_att)

        self.train_feature = torch.from_numpy(train_fea).float()
        self.test_seen_feature = torch.from_numpy(test_seen_fea).float()
        self.test_unseen_feature = torch.from_numpy(test_unseen_fea).float()

        matcontent = loadmat('data/CUB_200_2011/mat/label.mat')

        train_idx = matcontent['train_idx'] - 1
        train_label = matcontent['train_label_new']
        test_unseen_idex = matcontent['test_unseen_idex'] - 1
        test_seen_idex = matcontent['test_seen_idex'] - 1
        self.train_label = torch.from_numpy(train_idx.squeeze()).long()
        self.test_seen_label = torch.from_numpy(test_seen_idex.squeeze()).long()
        self.test_unseen_label = torch.from_numpy(test_unseen_idex.squeeze()).long()

        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)

    def read_imagenet(self, opt):
        with h5py.File('data/image_net/ILSVRC2012_res101_feature.mat', 'r') as f:
            labels = np.array(f['labels'], dtype=np.int64).flatten()
            labels_val = np.array(f['labels_val'], dtype=np.int64).flatten()

            train_fea = np.array(f['features']).T
            val_fea = np.array(f['features_val'])

        text_att = loadmat(f'{opt.imagenet_text_encode[:-5]}.mat')
        path = 'data/image_net/mat/text/WordEmbeddings_Lo_glove.840B.300d_ImageNet_trainval_classes_classes.pkl'
        # corre = pd.read_csv('data/image_net/wnid_correspondance.csv', sep=' ', names=['id', 'wnid'])
        splits = loadmat('data/image_net/ImageNet_splits.mat')

        seen = np.sort(splits['train_classes'].squeeze())
        unseen = np.sort(splits['val_classes'].squeeze())

        with open(path, 'rb') as f:
            data = pickle.load(f)

        all_train_classes = np.unique(splits['train_classes'])
        val_classes = np.unique(splits['val_classes'])
        train_seen_classes = set(data.keys()).intersection(all_train_classes)
        val_classes_with_aux = set(data.keys()).intersection(val_classes)
        predict_classes = val_classes_with_aux

        seen_classes = train_seen_classes
        unseen_classes = val_classes_with_aux

        train_seen_indices = select_indices(labels, seen_classes)
        val_seen_indices = select_indices(labels_val, seen_classes)
        val_unseen_indices = select_indices(labels_val, unseen_classes)

        train_fea = train_fea[train_seen_indices]
        val_unseen_fea = val_fea[val_unseen_indices]
        val_seen_fea = val_fea[val_seen_indices]

        # Image Features Normalization
        min_max_scaler = preprocessing.MinMaxScaler().fit(train_fea)
        self.train_feature = torch.from_numpy(min_max_scaler.transform(train_fea)).float()
        self.test_seen_feature = torch.from_numpy(min_max_scaler.transform(val_seen_fea)).float()
        self.test_unseen_feature = torch.from_numpy(min_max_scaler.transform(val_unseen_fea)).float()
        # self.train_feature = torch.from_numpy((train_fea)).float()
        # self.test_seen_feature = torch.from_numpy((val_seen_fea)).float()
        # self.test_unseen_feature = torch.from_numpy((val_unseen_fea)).float()

        self.train_label = torch.from_numpy(labels[train_seen_indices].squeeze()).long()
        self.test_seen_label = torch.from_numpy(labels_val[val_seen_indices].squeeze()).long()
        self.test_unseen_label = torch.from_numpy(labels_val[val_unseen_indices].squeeze()).long()

        self.seenclasses = torch.from_numpy(np.fromiter(seen_classes, int, len(seen_classes)))
        self.unseenclasses = torch.from_numpy(np.fromiter(unseen_classes, int, len(unseen_classes)))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        # self.train_class = self.seenclasses.clone()
        # self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        # self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        # self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)

        self.attribute = {}
        train_attribute = []
        for idx, (k, v) in enumerate(data.items()):
            self.attribute[k] = text_att['features'][idx]
            if k in seen_classes:
                train_attribute.append(k)

        self.train_att = text_att['train_att'].astype(np.float32)
        self.test_att = text_att['val_att'].astype(np.float32)
        self.attribute_to_idx = np.array(train_attribute)

        if opt.normalize_semantics:
            # Semantic Features Normalization
            min_max_scaler = preprocessing.MinMaxScaler().fit(self.train_att)
            # std_scaler = preprocessing.StandardScaler().fit(self.train_att)
            self.train_att = min_max_scaler .transform(self.train_att)
            self.test_att = min_max_scaler.transform(self.test_att)
        # np.array(list(self.attribute.keys()))


class FeatDataLayer(object):
    def __init__(self, label, feat_data, opt):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()
        self._epoch = 0

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def forward(self):
        new_epoch = False
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs

def get_text_feature(text_dir, train_test_split_dir):
    train_test_split = loadmat(train_test_split_dir)
    if isinstance(text_dir, np.ndarray):
        text_feature = text_dir
    else:
        text_feature = loadmat(text_dir)['PredicateMatrix']
    # get training text feature
    train_cid = train_test_split['train_cid'].squeeze()
    train_text_feature = text_feature[train_cid - 1]  # 0-based index

    # get testing text feature
    test_cid = train_test_split['test_cid'].squeeze()
    test_text_feature = text_feature[test_cid - 1]  # 0-based index
    return train_text_feature.astype(np.float32), test_text_feature.astype(np.float32)
