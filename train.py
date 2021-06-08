import torch
import wandb
from transform import Flow
from toydata import ToyData
from torch.utils.data import DataLoader, random_split
from utils import make_toy_graph
from torchviz import make_dot
import pyro
import matplotlib.pyplot as plt
import tqdm
from distributions import DoubleDistribution, StandardNormal, Normal, SemanticDistribution
import affine_coupling
from permuters import LinearLU, Permuter, Reverse
import torch.nn as nn
from affine_coupling import AffineCoupling
import torch.optim as optim
import torch.distributions as dist
from act_norm import ActNormBijection
from text_encoders.text_encoder import AlbertEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = wandb.init(project='toy_data_zf', entity='mvalente',
                 config=r'config/base_conf.yaml')

config = wandb.config

points_per_sample = 30000

input_dim = 4
context_dim = 2
split_dim = input_dim - context_dim

train_loader = DataLoader(,
                          batch_size=config['batch_size'],
                          shuffle=False, pin_memory=True)


visual_distribution = dist.MultivariateNormal(torch.zeros(split_dim).to(device), torch.eye(split_dim).to(device))
semantic_distribution = SemanticDistribution(contexts, torch.ones(context_dim).to(device), (2, 1))

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
    transforms.append(AffineCoupling(input_dim, hidden_dims=[2], non_linearity=non_linearity, net=config['net']))

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
        centralizing_loss = flow.centralizing_loss(data, targets, cs.to(device)) * config['wt_c_l']
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
