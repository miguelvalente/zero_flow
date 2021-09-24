import numpy as np
import pickle
import scipy.io as sio
from sklearn import preprocessing
import torch

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.dataset in ['FLO', 'cub2011']:
            if 'easy' or 'hard' in opt.split:
                self.read_easy(opt)
            else:
                self.read(opt)
        else:
            self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        # for i in range(self.seenclasses.shape[0]):
        #     self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

    def read_easy(self, opt):
        txt_feat_path = 'data/CIZSL/CUB2011/CUB_Porter_7551D_TFIDF_new.mat'
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")
        is_val = False
        if 'easy' in opt.split:
            train_test_split_dir = 'data/CIZSL/CUB2011/train_test_split_easy.mat'
            pfc_label_path_train = 'data/CIZSL/CUB2011/labels_train.pkl'
            pfc_label_path_test = 'data/CIZSL/CUB2011/labels_test.pkl'
            pfc_feat_path_train = 'data/CIZSL/CUB2011/pfc_feat_train.mat'  # pfc_feat (8855, 3584)
            pfc_feat_path_test = 'data/CIZSL/CUB2011/pfc_feat_test.mat'  # pfc_feat (2933, 3584)

            if is_val:
                train_cls_num = 120
                test_cls_num = 30
            else:
                train_cls_num = 150
                test_cls_num = 50
        elif 'hard' in opt.split:
            train_test_split_dir = 'data/CIZSL/CUB2011/train_test_split_hard.mat'
            pfc_label_path_train = 'data/CIZSL/CUB2011/labels_train_hard.pkl'
            pfc_label_path_test = 'data/CIZSL/CUB2011/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/CIZSL/CUB2011/pfc_feat_train_hard.mat'  # pfc_feat (9410, 3584)
            pfc_feat_path_test = 'data/CIZSL/CUB2011/pfc_feat_test_hard.mat'  # pfc_feat (2378, 3584)

            if is_val:
                train_cls_num = 130
                test_cls_num = 30
            else:
                train_cls_num = 160
                test_cls_num = 40

        if is_val:
            # load .mat as numpy by scipy.io as sio, easy (8855, 3584) e.g.
            data_features = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
            # load .pkl label file
            with open(pfc_label_path_train) as fout:
                data_labels = pickle.load(fout, encoding="latin1")
                # print('=============', data_labels.shape)
            # assert 0

            # why we use data_labels < train_cls_num
            self.pfc_feat_data_train = data_features[data_labels < train_cls_num]
            self.pfc_feat_data_test = data_features[data_labels >= train_cls_num]
            self.labels_train = data_labels[data_labels < train_cls_num]
            self.labels_test = data_labels[data_labels >= train_cls_num] - train_cls_num

            text_features, _ = get_text_feature(txt_feat_path, train_test_split_dir)  # Z_tr, Z_te
            self.train_text_feature, self.test_text_feature = text_features[:train_cls_num], text_features[train_cls_num:]
            self.att_dim = self.train_text_feature.shape[1]
        else:
            self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
            self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
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

        if not is_val:
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

    def read(self, opt):
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/data.mat")

        matcontent = sio.loadmat(opt.data_dir)

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
        test_seen_feature = min_max_scaler .transform(test_seen_fea)
        test_unseen_feature = min_max_scaler .transform(test_unseen_fea)

        if opt.normalize_semantics:
            # Semantic Features Normalization
            min_max_scaler = preprocessing.MinMaxScaler().fit(self.train_att)
            # std_scaler = preprocessing.StandardScaler().fit(self.train_att)
            self.train_att = min_max_scaler .transform(self.train_att)
            self.test_att = min_max_scaler.transform(self.test_att)

        # scaler = preprocessing.MinMaxScaler()
        # _train_feature = scaler.fit_transform(train_fea)
        # _test_seen_feature = scaler.transform(test_seen_fea)
        # _test_unseen_feature = scaler.transform(test_unseen_fea)
        # mx = _train_feature.max()
        # train_fea = train_fea * (1 / mx)
        # test_seen_fea = test_seen_fea * (1 / mx)
        # test_unseen_fea = test_unseen_fea * (1 / mx)

        self.train_feature = torch.from_numpy(train_fea).float()
        self.test_seen_feature = torch.from_numpy(test_seen_fea).float()
        self.test_unseen_feature = torch.from_numpy(test_unseen_fea).float()

        matcontent = sio.loadmat('data/CUB_200_2011/mat/label.mat')

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

def get_text_feature(dir, train_test_split_dir):
    train_test_split = sio.loadmat(train_test_split_dir)
    # get training text feature
    train_cid = train_test_split['train_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    train_text_feature = text_feature[train_cid - 1]  # 0-based index

    # get testing text feature
    test_cid = train_test_split['test_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    test_text_feature = text_feature[test_cid - 1]  # 0-based index
    return train_text_feature.astype(np.float32), test_text_feature.astype(np.float32)

