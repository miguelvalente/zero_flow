import os
import torch
import wandb
from transform import Flow
import tqdm
from distributions import DoubleDistribution, SemanticDistribution
from permuters import LinearLU, Permuter, Reverse
import torch.nn as nn
from affine_coupling import AffineCoupling
import torch.optim as optim
import torch.distributions as dist
from act_norm import ActNormBijection
from text_encoders.text_encoder import ProphetNet
from text_encoders.context_encoder import ContextEncoder
from cub2011 import Cub2011

import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from convert import CostumTransform
import torchvision.datasets as datasets
import torchvision.transforms as transforms


#CUDA_LAUNCH_BLOCKING = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = wandb.init(project='zero_flow_CUB', entity='mvalente',
                 config=r'config/base_conf.yaml')

config = wandb.config

normalize_cub = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0 / 255, 1.0 / 255, 1.0 / 255])

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize_cub,
    transforms.ToPILImage(mode='RGB'),
    CostumTransform(config['image_encoder'])
])

cub = Cub2011(root='/project/data/', transform=transforms, download=False)
seen_id = list(set(cub.data['target']))
unseen_id = list(set(cub.data_unseen['target']))

context_encoder = ContextEncoder(config, seen_id, unseen_id, device)
contexts = context_encoder.contexts.to(device)
cs = context_encoder.cs.to(device)
cu = context_encoder.cu.to(device)
# cs = context_encoder.cs
# cu = context_encoder.cu
# normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                           std=[0.229, 0.224, 0.225])

# train_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(config['image_net_dir'], transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize_imagenet,
#         transforms.ToPILImage(mode='RGB'),
#         CostumTransform(config['image_encoder'])
#     ])), batch_size=config['batch_size'], shuffle=False, pin_memory=True)

train_loader = torch.utils.data.DataLoader(cub, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

input_dim = cub[0][0].shape.numel()
context_dim = contexts[0].shape.numel()
split_dim = input_dim - context_dim

semantic_distribution = SemanticDistribution(contexts, torch.ones(context_dim).to(device), (context_dim, 1))
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

transforms = []
for index in range(config['block_size']):
    if config['act_norm']:
        transforms.append(ActNormBijection(input_dim, data_dep_init=True))
    transforms.append(permuter(input_dim))
    transforms.append(AffineCoupling(input_dim, hidden_dims=[context_dim], non_linearity=non_linearity, net=config['net']))

flow = Flow(transforms, base_dist)
flow.train()
flow = flow.to(device)

print(f'Number of trainable parameters: {sum([x.numel() for x in flow.parameters()])}')
run.watch(flow)
optimizer = optim.Adam(flow.parameters(), lr=config['lr'])

epochs = tqdm.trange(1, config['epochs'])

for epoch in epochs:
    losses = []
    losses_flow = []
    losses_centr = []
    losses_mmd = []
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss_flow = - flow.log_prob(data, targets).mean() * config['wt_f_l']
        centralizing_loss = flow.centralizing_loss(data, targets, cs) * config['wt_c_l']
        mmd_loss = flow.mmd_loss(data, cu.to(device)) * config['wt_mmd_l']
        loss = loss_flow + centralizing_loss + mmd_loss
        loss.backward()
        optimizer.step()

        losses_flow.append(loss_flow.item())
        losses_centr.append(centralizing_loss.item())
        losses_mmd.append(mmd_loss.item())
        losses.append(loss.item())

    run.log({"loss": sum(losses) / len(losses),
             "loss_flow": sum(losses_flow) / len(losses_flow),  # }, step=epoch)
             "loss_central": sum(losses_centr) / len(losses_centr),  # }, step=epoch)
             "loss_mmd": sum(losses_mmd) / len(losses_mmd)}, step=epoch)

    if loss.isnan():
        print('Nan in loss!')
        Exception('Nan in loss!')
