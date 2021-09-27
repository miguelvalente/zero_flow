import argparse
import glob
import json
import math
import os
import random
from time import gmtime, strftime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import yaml
from scipy.io import loadmat, savemat
from sklearn.metrics.pairwise import cosine_similarity

import classifier
import wandb
from act_norm import ActNormBijection
from affine_coupling import AffineCoupling
from dataloaders.dataset_GBU import DATA_LOADER, FeatDataLayer
from distributions import DoubleDistribution, SemanticDistribution
from permuters import LinearLU, Permuter, Reverse
from transform import Flow
from utils import Result, log_print, save_model, synthesize_feature

CUDA_LAUNCH_BLOCKING = 1
os.environ['WANDB_MODE'] = 'online'
run = wandb.init(project='zero_flow_imagenet', entity='mvalente',
                 config=r'config/flow.yaml')

config = wandb.config
# with open(f"{config.data_dir[:-3]}yaml", 'r') as y:
#     temp = yaml.safe_load(y)
#     wandb.config['image_encoder'] = 'cizsl'
#     wandb.config['text_encoder'] = 'cizsl'
#     # wandb.config['image_encoder'] = 'cizsl' temp['image_encoder']
#     # wandb.config['text_encoder'] = 'cizsl'temp['text_encoder']
#     # wandb.config['split'] = temp['split']
#     # wandb.config['split'] = 'easy'
#     wandb.config['dataset'] = temp['dataset']
#     del temp

wandb.config['dataset'] = 'imagenet'
wandb.config['image_encoder'] = 'resnet101'
wandb.config['text_encoder'] = 'albertxxl'

wandb.define_metric('Harmonic Mean', summary='max')
wandb.define_metric('Accuracy Unseen', summary='max')
wandb.define_metric('Accuracy Seen', summary='max')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if config.manualSeed is None:
    config.manualSeed = random.randint(1, 10000)
print("Random Seed: ", config.manualSeed)
np.random.seed(config.manualSeed)
random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)
torch.cuda.manual_seed_all(config.manualSeed)
cudnn.benchmark = True


def train():
    dataset = DATA_LOADER(config)
    config.C_dim = dataset.att_dim
    config.X_dim = dataset.feature_dim
    config.y_dim = dataset.ntrain_class
    out_dir = 'out/{}/mask-{}_pi-{}_c-{}_ns-{}_wd-{}_lr-{}_nS-{}_bs-{}_ps-{}'.format(
        config.dataset, config.dropout, config.pi, config.prototype,
        config.number_sample, config.weight_decay, config.lr, config.number_sample, config.batchsize, config.pi)

    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    log_dir = out_dir + '/log_{}.txt'.format(config.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), config)
    config.niter = int(dataset.ntrain / config.batchsize) * config.epochs

    result = Result()
    sim = cosine_similarity(dataset.train_att, dataset.train_att)
    min_idx = np.argmin(sim.sum(-1))
    min = dataset.train_att[min_idx]
    max_idx = np.argmax(sim.sum(-1))
    max = dataset.train_att[max_idx]
    medi_idx = np.argwhere(sim.sum(-1) == np.sort(sim.sum(-1))[int(sim.shape[0] / 2)])
    medi = dataset.train_att[int(medi_idx)]
    vertices = torch.from_numpy(np.stack((min, max, medi))).float().cuda()

    input_dim = dataset.train_feature.shape[1]
    if config.relative_positioning:
        context_dim = dataset.train_att.shape[1] if config.semantic_vector_dim == 0 else config.semantic_vector_dim
    else:
        context_dim = dataset.train_att.shape[1]
    split_dim = input_dim - context_dim

    semantic_distribution = SemanticDistribution(torch.tensor(dataset.train_att).cuda(), torch.ones(context_dim).cuda())
    visual_distribution = dist.MultivariateNormal(torch.zeros(split_dim).cuda(), torch.eye(split_dim).cuda())
    base_dist = DoubleDistribution(visual_distribution, semantic_distribution, input_dim, context_dim)

    permuter = lambda dim: LinearLU(num_features=dim, eps=1.0e-5)
    non_linearity = nn.PReLU(init=0.01)
    hidden_dims = [input_dim] * 3

    transform = []
    for index in range(5):
        if True:
            transform.append(ActNormBijection(input_dim, data_dep_init=True))
        transform.append(permuter(input_dim))
        transform.append(AffineCoupling(input_dim, hidden_dims, non_linearity=non_linearity))

    flow = Flow(transform, base_dist).cuda()

    if config.relative_positioning:
        sm = GSModule(vertices, context_dim).cuda()
        parameters = list(flow.parameters()) + list(sm.parameters())
    else:
        parameters = list(flow.parameters())
    print(flow)

    optimizer = optim.Adam(parameters,
                           lr=float(config.lr),
                           weight_decay=float(config.weight_decay))
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
        if config.relative_positioning:
            sm.zero_grad()

        loss = 0
        blobs = data_layer.forward()
        feat_data = blobs['data']  # image data
        labels_numpy = blobs['labels'].astype(int)  # class labels
        labels = torch.from_numpy(labels_numpy.astype('int')).cuda()

        if config.dataset == 'cub2011':
            if config.relative_positioning:
                C = np.array([dataset.train_att[i, :] for i in labels])
                # C = np.array([dataset.train_att_sampled[i] for i in blobs['idx']])
                C = torch.from_numpy(C.astype('float32')).cuda()
            else:
                C = np.array([dataset.train_att_sampled[i] for i in blobs['idx']])
                C = torch.from_numpy(C.astype('float32')).cuda()
        else:
            labels_d = np.array((labels.detach().cpu()))
            C = torch.stack([dataset.attribute[i] for i in labels_d]).cuda()

        X = torch.from_numpy(feat_data).cuda()

        if config.relative_positioning:
            sr = sm(C)
        else:
            sr = labels
        z = config.pi * torch.randn(config.batchsize, input_dim).cuda()
        mask = torch.cuda.FloatTensor(input_dim).uniform_() > config.dropout
        z = mask * z
        X = X + z

        if 'IZF' in config.loss_type:
            log_prob, ldj = flow.log_prob(X, sr)
            log_prob += ldj
            loss_flow = - log_prob.mean() * config.flow_loss
        else:
            log_prob, ldj = flow.log_prob(X, sr)
            # z_, log_jac_det = flow(X, sr)
            # loss = torch.mean(z_**2) / 2 - torch.mean(log_jac_det) / 2048
            loss_flow = torch.mean(log_prob**2) / 2 - torch.mean(ldj) / 2048

        loss += loss_flow
        run.log({"loss_flow": loss_flow.item()})

        if config.relative_positioning:
            with torch.no_grad():
                sr = sm(torch.from_numpy(dataset.train_att).cuda())
        else:
            sr = torch.from_numpy(dataset.train_att).cuda()

        if config.centralizing_loss > 0:
            if config['dataset'] == 'cub2011':
                centralizing_loss = flow.centralizing_loss(X, labels, sr.cuda(), torch.unique(dataset.test_seen_label))
            else:
                centralizing_loss = flow.centralizing_loss(X, labels, sr.cuda(), torch.unique(dataset.test_seen_label), dataset.attribute_to_idx)
            loss += centralizing_loss
            run.log({"loss_central": centralizing_loss.item()})

        if config.prototype > 0:
            if config.relative_positioning:
                with torch.no_grad():
                    sr = sm(torch.from_numpy(dataset.train_att).cuda())
            else:
                sr = torch.from_numpy(dataset.train_att).cuda()

            x_ = flow.generation(torch.cat((sr, visual_distribution.sample((sr.shape[0],))), dim=1))
            prototype_loss = config.prototype * mse(x_, x_mean)
            loss += prototype_loss
            run.log({"loss_proto": prototype_loss.item()})

        run.log({"loss": loss.item()})

        loss.backward()
        nn.utils.clip_grad_value_(flow.parameters(), 1.0)
        optimizer.step()
        if it % iters == 0:
            lr_scheduler.step()
        if it % config.disp_interval == 0 and it:
            log_text = f'Iter-[{it}/{config.niter}]; loss: {loss.item():.3f}'
            log_print(log_text, log_dir)

        if it % config.evl_interval == 0 and it >= 100:
            flow.eval()
            if config.relative_positioning:
                sm.eval()
                gen_feat, gen_label = synthesize_feature(flow, dataset, config, sm)
            else:
                gen_feat, gen_label = synthesize_feature(flow, dataset, config)

            # """ GZSL """
            if config.gzsl:
                train_X = torch.cat((dataset.train_feature, gen_feat), 0)
                train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)

                cls = classifier.CLASSIFIER(run, config, train_X, train_Y, dataset, dataset.test_seen_feature, dataset.test_unseen_feature,
                                            dataset.ntrain_class + dataset.ntest_class, True, config.classifier_lr, 0.5, 30, 3000, config.gzsl)

                result.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

                log_print("GZSL Softmax:", log_dir)
                log_print(f"U->T {cls.acc_unseen:.2f}  S->T {cls.acc_seen:.2f}  H {cls.H:.2f}"
                          f" Best_H [{result.best_acc_U_T:.2f} {result.best_acc_S_T:.2f} {result.best_acc:.2f}]"
                          f"| Iter-{result.best_iter}]", log_dir)

                if result.save_model:
                    files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    save_model(it, flow, sm, config.manualSeed, log_text,
                               f"{out_dir}/Best_model_GZSL_H_{result.best_acc:.2f}_S_{result.best_acc_S_T:.2f}_U_{result.best_acc_U_T:.2f}.tar")
            # """ ZSL """
            else:
                train_X = gen_feat
                train_Y = gen_label
                cls = classifier.CLASSIFIER(run, config, train_X, train_Y, dataset, dataset.test_unseen_feature, dataset.test_unseen_feature,
                                            dataset.ntest_class, True, config.classifier_lr, 0.5, 30, 3000, config.gzsl)

                result.update(it, cls.acc)
                log_print("ZSL Softmax:", log_dir)
                log_print(f"Best_Accuracy [{result.best_acc:.2f} | Iter-{result.best_iter}]", log_dir)

                if result.save_model:
                    files2remove = glob.glob(out_dir + '/Best_model_ZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    save_model(it, flow, sm, config.manualSeed, log_text,
                               f"{out_dir}/Best_model_ZSL_Acc_{result.best_acc:.2f}_.tar")

            flow.train()
            if config.relative_positioning:
                sm.train()
                if it % config.save_interval == 0 and it:
                    save_model(it, flow, sm, config.manualSeed, log_text,
                               out_dir + '/Iter_{:d}.tar'.format(it))
                    print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))
            else:
                if it % config.save_interval == 0 and it:
                    save_model(it, flow, 0, config.manualSeed, log_text,
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
