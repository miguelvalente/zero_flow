import csv
import pickle

import numpy as np
import torch
import torch.nn.init as init


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

    def update(self, it, acc):
        self.acc_list += [acc]
        self.iter_list += [it]
        self.save_model = False
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            self.save_model = True

    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.iter_list += [it]
        self.save_model = False
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U_T = acc_u
            self.best_acc_S_T = acc_s
            self.save_model = True

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_except_batch(x, num_dims=1):
    '''
    Averages all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_mean: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).mean(-1)

def reduce_mean_masked(x, mask, axis):
    x = x * mask.float()
    m = x.sum(axis=axis) / mask.sum(axis=axis).float()
    return m

def reduce_sum_masked(x, mask, axis):
    x = x * mask
    m = x.sum(axis=axis)
    return m

if __name__ == '__main__':
    x = torch.rand((2, 10, 10))
    mask = torch.ones(2, 10, 10)
    reduce_sum_masked(x, mask, axis=1)


def log_print(s, log):
    print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')


def synthesize_feature(flow, sm, dataset, opt):
    gen_feat = torch.FloatTensor(dataset.ntest_class * opt.number_sample, opt.X_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for i in range(dataset.ntest_class):
            text_feat = np.tile(dataset.test_att[i].astype('float32'), (opt.number_sample, 1))
            text_feat = torch.from_numpy(text_feat).cuda()
            sr = sm(text_feat)
            z = torch.randn(opt.number_sample, 1024).cuda()
            # z = z*z.norm(dim=-1, keepdim=True)
            G_sample = flow.generation(torch.cat((sr, z), dim=1))
            # G_sample = flow.reverse_sample(z, sr)
            gen_feat[i * opt.number_sample:(i + 1) * opt.number_sample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([opt.number_sample]) * i))
    return gen_feat, torch.from_numpy(gen_label.astype(int))


def save_model(it, flow, gs, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_flow': flow.state_dict(),
        'state_dict_semantic': gs.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)
