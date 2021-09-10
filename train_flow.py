import math
import os

import numpy as np
import timm
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import yaml
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import wandb
from act_norm import ActNormBijection
from affine_coupling import AffineCoupling
from convert import VisualExtractor
from dataloaders.cub2011 import Cub2011
from distributions import DoubleDistribution, SemanticDistribution
from nets import GSModule

from permuters import LinearLU, Permuter, Reverse
from text_encoders.context_encoder import ContextEncoder
from transform import Flow

# CUDA_LAUNCH_BLOCKING = 1
SAVE_PATH = 'checkpoints/'
os.environ['WANDB_MODE'] = 'online'
save = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = wandb.init(project='zero_flow_CUB_2.0', entity='mvalente',
                 config=r'config/flow.yaml')

with open('config/dataloader.yaml', 'r') as d, open('config/context_encoder.yaml', 'r') as c:
    wandb.config.update(yaml.safe_load(d))
    wandb.config.update(yaml.safe_load(c))

config = wandb.config

transforms_cub = None
# if config['image_encoder']:
#     transforms_cub = transforms.Compose([
#         VisualExtractor(config['image_encoder'])
#     ])

cub_train = Cub2011(config=config, which_split='train', root='/project/data/', transform=transforms_cub)
seen_id = cub_train.seen_id
unseen_id = cub_train.unseen_id

context_encoder = ContextEncoder(config, device=device)
contexts = context_encoder.contexts.to(device)
cs = contexts[seen_id]
cu = contexts[unseen_id]

train_loader = torch.utils.data.DataLoader(cub_train, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

cub_val = Cub2011(config=config, which_split='val', root='/project/data/', transform=transforms_cub)
val_loader = torch.utils.data.DataLoader(cub_val, batch_size=1000, shuffle=True, pin_memory=True)
test_id = cub_val.test_id

input_dim = cub_train[0][0].shape.numel()
context_dim = 1024  # contexts[0].shape.numel()
split_dim = input_dim - context_dim

semantic_distribution = SemanticDistribution(contexts, torch.ones(context_dim).to(device))
visual_distribution = dist.MultivariateNormal(torch.zeros(split_dim).to(device), torch.eye(split_dim).to(device))
base_dist = DoubleDistribution(visual_distribution, semantic_distribution, input_dim, context_dim)

if config['permuter'] == 'random':
    permuter = lambda dim: Permuter(permutation=torch.randperm(dim, dtype=torch.long).to(device))
elif config['permuter'] == 'reverse':
    permuter = lambda dim: Reverse(dim_size=dim)
elif config['permuter'] == 'manual':
    permuter = lambda dim: Permuter(permutation=torch.tensor([2, 3, 0, 1], dtype=torch.long).to(device))
elif config['permuter'] == 'LinearLU':
    permuter = lambda dim: LinearLU(num_features=dim, eps=1.0e-5)

if config['non_linearity'] == 'relu':
    non_linearity = torch.nn.ReLU()
elif config['non_linearity'] == 'prelu':
    non_linearity = nn.PReLU(init=0.01)
elif config['non_linearity'] == 'leakyrelu':
    non_linearity = nn.LeakyReLU()

if not config['hidden_dims']:
    hidden_dims = [input_dim // 2]
else:
    hidden_dims = config['hidden_dims']

transform = []
for index in range(config['block_size']):
    if config['act_norm']:
        transform.append(ActNormBijection(input_dim, data_dep_init=True))
    transform.append(permuter(input_dim))
    transform.append(AffineCoupling(
        input_dim,
        split_dim,
        context_dim=context_dim, hidden_dims=hidden_dims, non_linearity=non_linearity, net=config['net']))

model = Flow(transform, base_dist)
model.train()
model = model.to(device)

if config['relative_positioning']:
    contexts_copy = contexts.clone().detach().cpu().numpy()
    sim = cosine_similarity(contexts_copy, contexts_copy)
    min_idx = np.argmin(sim.sum(-1))
    minimum = contexts_copy[min_idx]
    max_idx = np.argmax(sim.sum(-1))
    maximum = contexts_copy[max_idx]
    medi_idx = np.argwhere(sim.sum(-1) == np.sort(sim.sum(-1))[int(sim.shape[0] / 2)])
    medi = contexts_copy[int(medi_idx)]
    vertices = torch.from_numpy(np.stack((minimum, maximum, medi))).float().cuda()
    sm = GSModule(vertices, 1024).cuda()
    parameters = list(model.parameters()) + list(sm.parameters())
else:
    parameters = list(model.parameters())

print(f'Number of trainable parameters: {sum([x.numel() for x in parameters])}')
# run.watch(model)
optimizer = optim.Adam(parameters, lr=config['lr'])

x_mean = cub_train.visual_features.mean(axis=0).cuda()
mse = nn.MSELoss()

for epoch in range(1, config['epochs']):
    losses = []
    for data, targets, _ in tqdm.tqdm(train_loader, desc=f'Epoch({epoch})'):
        data = data.to(device)
        targets = targets.to(device)
        model.zero_grad()

        if config['relative_positioning']:
            sm.zero_grad()
            relative_contexts = torch.stack([contexts[i, :] for i in targets])
            relative_contexts = sm(relative_contexts)
            cs_relative = sm(cs)
            cu_relative = sm(cu)
            log_prob, ldj = model.log_prob(data, relative_contexts)
            centralizing_loss = model.centralizing_loss(data, targets, cs_relative, seen_id) * config['wt_c_l']
            mmd_loss = model.mmd_loss(data, cu_relative) * config['wt_mmd_l']
        else:
            log_prob, ldj = model.log_prob(data, targets)
            centralizing_loss = model.centralizing_loss(data, targets, cs, seen_id) * config['wt_c_l']
            mmd_loss = model.mmd_loss(data, cu) * config['wt_mmd_l']

        if config['loss_type'] == 'IZF':
            log_prob += ldj
            loss_flow = - log_prob.mean() * config['wt_f_l']
        else:
            loss_flow = torch.mean(log_prob**2) / 2 - torch.mean(ldj) / 2048

        loss = loss_flow + centralizing_loss + mmd_loss
        loss.backward()

        if config['wt_p_l'] > 0.0:
            with torch.no_grad():
                sr = sm(torch.cat((cs, cu), dim=0))
            gens = model.generation(F.pad(sr, (0, 1024))).cuda()
            x_ = model.generation(gens)
            prototype_loss = config['wt_p_l'] * mse(x_, x_mean)
            prototype_loss.backward()

        # nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

        if loss.isnan():
            print('Nan in loss!')
            Exception('Nan in loss!')

        losses.append(loss.item())

        run.log({"loss": loss.item(),
                 "loss_flow": loss_flow.item(),
                 "loss_central": centralizing_loss.item(),
                 "loss_mmd": mmd_loss.item()})  # "loss_proto": prototype_loss.item()})  # "loss_mmd": mmd_loss.item(),

    if False:
        with torch.no_grad():
            model.eval()
            sm.eval()
            loss_flow_val = 0
            centralizing_loss_val = 0
            mmd_loss_val = 0
            loss_val = 0
            prototype_loss = 0

            losses_val = []
            for data_val, targets_val, _ in tqdm.tqdm(val_loader, desc=f'Validation Epoch({epoch})'):
                data_val = data_val.to(device)
                targets_val = targets_val.to(device)

                if config['relative_positioning']:
                    relative_contexts = torch.stack([contexts[i, :] for i in targets])
                    relative_contexts = sm(relative_contexts)
                    cs_relative = sm(cs)
                    cu_relative = sm(cu)
                    log_prob, _ = model.log_prob(data, relative_contexts)
                    centralizing_loss_val = model.centralizing_loss(data, targets, cs_relative, seen_id) * config['wt_c_l']
                else:
                    log_prob, _, _ = model.log_prob(data, targets)
                    centralizing_loss_val = model.centralizing_loss(data, targets, cs, seen_id) * config['wt_c_l']

                loss_flow_val = - log_prob.mean() * config['wt_f_l']
                centralizing_loss_val = model.centralizing_loss(data_val, targets_val, cs, test_id) * config['wt_c_l']
                # mmd_loss_val = model.mmd_loss(data_val, cu) * config['wt_mmd_l']
                loss_val = loss_flow + centralizing_loss  # + mmd_loss
                losses_val.append(loss_val.item())

        run.log({"loss_val": sum(losses_val) / len(losses_val)})
    run.log({"epoch": epoch})
    print(f'Epoch({epoch}): loss:{sum(losses)/len(losses)}')

    model.train()
    sm.train()
    if epoch % 20 == 0:
        state = {'config': config.as_dict(),
                 'split': config['split'],
                 'state_dict_flow': model.state_dict(),
                 'state_dict_sm': sm.state_dict(),
                 'optimizer': optimizer.state_dict()}

        torch.save(state, f'{SAVE_PATH}{wandb.run.name}-{epoch}.pth')

    if loss.isnan():
        print('Nan in loss!')
        Exception('Nan in loss!')
