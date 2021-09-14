import os
import math
import glob
import json
import random
import argparse
import classifier
from utils import Result, synthesize_feature, save_model, log_print
import torch.nn as nn
import torch.optim as optim
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn.functional as F
from time import gmtime, strftime
import torch.backends.cudnn as cudnn
from dataloaders.dataset_GBU import FeatDataLayer, DATA_LOADER
from sklearn.metrics.pairwise import cosine_similarity

from transform import Flow
from distributions import SemanticDistribution, DoubleDistribution
from permuters import LinearLU, Permuter, Reverse
from act_norm import ActNormBijection
from affine_coupling import AffineCoupling
import torch.distributions as dist
import wandb
import yaml
import tqdm
import numpy as np
import torch


run = wandb.init(project='zero_flow_CUB_2.0', entity='mvalente',
                 config=r'config/flow.yaml')

with open('config/classifier.yaml', 'r') as c:
    wandb.config.update(yaml.safe_load(c))

config = wandb.config

# image_embedding', default='res101', type=str)
#   desc:
#   value:, default='data/data', help='path to dataset')
# class_embedding', default='att', type=str)
#   desc:
#   value:, default='data/data', help='path to dataset')

os.environ['CUDA_VISIBLE_DEVICES'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if config.manualSeed is None:
    config.manualSeed = random.randint(1, 10000)
print("Random Seed: ", config.manualSeed)
np.random.seed(config.manualSeed)
random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)
torch.cuda.manual_seed_all(config.manualSeed)

cudnn.benchmark = True
print('Running parameters:')
print(json.dumps(vars(config), indent=4, separators=(',', ': ')))


def train():
    dataset = DATA_LOADER(config)
    config.C_dim = dataset.att_dim
    config.X_dim = dataset.feature_dim
    config.y_dim = dataset.ntrain_class
    out_dir = 'out/{}/mask-{}_pi-{}_c-{}_ns-{}_wd-{}_lr-{}_nS-{}_bs-{}_ps-{}'.format(
        config.dataset, config.dropout, config.pi, config.prototype, config.nSample, config.weight_decay, config.lr, config.nSample, config.batchsize, config.pi)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    log_dir = out_dir + '/log_{}.txt'.format(config.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), config)
    config.niter = int(dataset.ntrain / config.batchsize) * config.gen_nepoch

    result_gzsl_soft = Result()
    sim = cosine_similarity(dataset.train_att, dataset.train_att)
    min_idx = np.argmin(sim.sum(-1))
    min = dataset.train_att[min_idx]
    max_idx = np.argmax(sim.sum(-1))
    max = dataset.train_att[max_idx]
    medi_idx = np.argwhere(sim.sum(-1) == np.sort(sim.sum(-1))[int(sim.shape[0] / 2)])
    medi = dataset.train_att[int(medi_idx)]
    vertices = torch.from_numpy(np.stack((min, max, medi))).float().cuda()

    input_dim = 2048
    context_dim = config['semantic_vector_dim']  # contexts[0].shape.numel()
    split_dim = input_dim - context_dim

    semantic_distribution = SemanticDistribution(torch.tensor(dataset.train_att).cuda(), torch.ones(context_dim).cuda())
    visual_distribution = dist.MultivariateNormal(torch.zeros(split_dim).cuda(), torch.eye(split_dim).cuda())
    base_dist = DoubleDistribution(visual_distribution, semantic_distribution, input_dim, context_dim)

    permuter = lambda dim: LinearLU(num_features=dim, eps=1.0e-5)

    non_linearity = nn.PReLU(init=0.01)

    hidden_dims = [input_dim // 2]

    transform = []
    for index in range(5):
        if True:
            transform.append(ActNormBijection(input_dim, data_dep_init=True))
        transform.append(permuter(input_dim))
        transform.append(AffineCoupling(
            input_dim,
            split_dim,
            context_dim=context_dim, hidden_dims=hidden_dims, non_linearity=non_linearity, net='MLP'))

    flow = Flow(transform, base_dist).cuda()

    sm = GSModule(vertices, int(config['semantic_vector_dim')).cuda()
    print(flow)
    optimizer = optim.Adam(list(flow.parameters()) + list(sm.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.8, step_size=15)

    mse = nn.MSELoss()
    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('GSMFlow Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    start_step = 0
    prototype_loss = 0

    x_mean = torch.from_numpy(dataset.tr_cls_centroid).cuda()
    iters = math.ceil(dataset.ntrain / config.batchsize)
    for it in tqdm.tqdm(range(start_step, config.niter + 1), desc='Iterations'):
        flow.zero_grad()
        sm.zero_grad()

        blobs = data_layer.forward()
        feat_data = blobs['data']  # image data
        labels_numpy = blobs['labels'].astype(int)  # class labels
        labels = torch.from_numpy(labels_numpy.astype('int')).cuda()

        C = np.array([dataset.train_att[i, :] for i in labels])
        C = torch.from_numpy(C.astype('float32')).cuda()
        X = torch.from_numpy(feat_data).cuda()

        sr = sm(C)
        z = config.pi * torch.randn(config.batchsize, 2048).cuda()
        mask = torch.cuda.FloatTensor(2048).uniform_() > config.dropout
        z = mask * z
        X = X + z

        log_prob, ldj = flow.log_prob(X, sr)
        log_prob += ldj
        loss_flow = - log_prob.mean() * 2
        with torch.no_grad():
            sr = sm(torch.from_numpy(dataset.train_att).cuda())
        centralizing_loss = flow.centralizing_loss(X, labels, sr.cuda(), torch.unique(dataset.test_seen_label))

        loss = loss_flow + centralizing_loss
        # z_, log_jac_det = flow(X, sr)

        # loss = torch.mean(z_**2) / 2 - torch.mean(log_jac_det) / 2048

        loss.backward(retain_graph=True)

        if config.prototype > 0.0:
            with torch.no_grad():
                sr = sm(torch.from_numpy(dataset.train_att).cuda())
            z = torch.zeros(dataset.ntrain_class, 2048).cuda()
            # x_ = flow.reverse_sample(z, sr)
            x_ = flow.generation(torch.cat((sr, visual_distribution.sample((sr.shape[0],))), dim=1))
            prototype_loss = config.prototype * mse(x_, x_mean)
            prototype_loss.backward()

        optimizer.step()
        if it % iters == 0:
            lr_scheduler.step()
        if it % config.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; loss: {:.3f}; prototype_loss:{:.3f};'.format(it, config.niter, loss.item(),
                                                                                   prototype_loss.item())
            log_print(log_text, log_dir)

        if it % config.evl_interval == 0 and it > 2000:
            flow.eval()
            sm.eval()
            gen_feat, gen_label = synthesize_feature(flow, sm, dataset, config)

            train_X = torch.cat((dataset.train_feature, gen_feat), 0)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)

            """ GZSL"""

            cls = classifier.CLASSIFIER(config, train_X, train_Y, dataset, dataset.test_seen_feature, dataset.test_unseen_feature,
                                        dataset.ntrain_class + dataset.ntest_class, True, config.classifier_lr, 0.5, 30, 3000, True)

            result_gzsl_soft.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

            log_print("GZSL Softmax:", log_dir)
            log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                cls.acc_unseen, cls.acc_seen, cls.H, result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

            if result_gzsl_soft.save_model:
                files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                save_model(it, flow, sm, config.manualSeed, log_text,
                           out_dir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl_soft.best_acc,
                                                                                              result_gzsl_soft.best_acc_S_T,
                                                                                              result_gzsl_soft.best_acc_U_T))

            sm.train()
            flow.train()
            if it % config.save_interval == 0 and it:
                save_model(it, flow, sm, config.manualSeed, log_text,
                           out_dir + '/Iter_{:d}.tar'.format(it))
                print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))

class LinearModule(nn.Module):
    def __init__(self, vertice, out_dim):
        super(LinearModule, self).__init__()
        self.register_buffer('vertice', vertice.clone())
        self.fc = nn.Linear(vertice.numel(), out_dim)

    def forward(self, semantic_vec):
        input_offsets = semantic_vec - self.vertice
        response = F.relu(self.fc(input_offsets))
        return response

class GSModule(nn.Module):
    def __init__(self, vertices, out_dim):
        super(GSModule, self).__init__()
        self.individuals = nn.ModuleList()
        assert vertices.dim() == 2, 'invalid shape : {:}'.format(vertices.shape)
        self.out_dim = out_dim
        self.require_adj = False
        for i in range(vertices.shape[0]):
            layer = LinearModule(vertices[i], out_dim)
            self.individuals.append(layer)

    def forward(self, semantic_vec):
        responses = [indiv(semantic_vec) for indiv in self.individuals]
        global_semantic = sum(responses)
        return global_semantic

if __name__ == "__main__":
    train()
