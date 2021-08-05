import os

import timm
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import yaml
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import wandb
from act_norm import ActNormBijection
from affine_coupling import AffineCoupling
from convert import VisualExtractor
from dataloaders.cub2011 import Cub2011
from distributions import DoubleDistribution, SemanticDistribution
from permuters import LinearLU, Permuter, Reverse
from text_encoders.context_encoder import ContextEncoder
from text_encoders.text_encoder import AlbertEncoder, ProphetNet
from transform import Flow

# CUDA_LAUNCH_BLOCKING = 1
SAVE_PATH = 'checkpoints/'
os.environ['WANDB_MODE'] = 'online'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = wandb.init(project='zero_flow_CUB_2.0', entity='mvalente',
                 config=r'config/flow.yaml')

with open('config/dataloader.yaml', 'r') as d, open('config/context_encoder.yaml', 'r') as c:
    wandb.config.update(yaml.safe_load(d))
    wandb.config.update(yaml.safe_load(c))

wandb.config['image_encoder'] = wandb.config['mat_file_visual'].split('/')[-1].split('.')[0]
config = wandb.config

if config['image_encoder'] != 'res101_ordered':
    transforms_cub = transforms.Compose([
        VisualExtractor(config['image_encoder'])
    ])
else:
    transforms_cub = None

cub_train = Cub2011(which_split='train', root='/project/data/', config=config, transform=transforms_cub)
seen_id = cub_train.seen_id
unseen_id = cub_train.unseen_id

context_encoder = ContextEncoder(config, device=device)
contexts = context_encoder.contexts.to(device)
cs = contexts[seen_id].to(device)
cu = contexts[unseen_id].to(device)

train_loader = torch.utils.data.DataLoader(cub_train, batch_size=config['batch_size_f'], shuffle=True, pin_memory=True)

cub_val = Cub2011(which_split='test', root='/project/data/', config=config, transform=transforms_cub)
val_loader = torch.utils.data.DataLoader(cub_val, batch_size=1000, shuffle=True, pin_memory=True)
test_id = cub_val.test_id

input_dim = 2048
context_dim = contexts[0].shape.numel()
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
    permuter = lambda dim: LinearLU(num_features=dim, eps=float(config['lu_eps']))

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

transforms = []
for index in range(config['block_size']):
    if config['act_norm']:
        transforms.append(ActNormBijection(input_dim, data_dep_init=True))
    transforms.append(permuter(input_dim))
    transforms.append(AffineCoupling(
        input_dim,
        split_dim,
        context_dim=context_dim, hidden_dims=hidden_dims, non_linearity=non_linearity, eps=float(config['aff_eps'])))

model = Flow(transforms, base_dist)
model.train()
model = model.to(device)

print(f'Number of trainable parameters: {sum([x.numel() for x in model.parameters()])}')
run.watch(model)
optimizer = optim.Adam(model.parameters(), lr=config['lr_f'])

for epoch in range(1, config['epochs_f']):
    losses = []
    loss = 0
    loss_flow = 0
    centralizing_loss = 0
    mmd_loss = 0
    for data, targets, _ in tqdm.tqdm(train_loader, desc=f'Epoch({epoch})'):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        # loss_flow, lg, ldj = - model.log_prob(data, targets).mean() * config['wt_f_l']
        log_prob, lg, ldj = model.log_prob(data, targets)
        ldj = ldj.mean()
        lg = lg.mean()

        loss_flow = - log_prob.mean() * config['wt_f_l']
        centralizing_loss = model.centralizing_loss(data, targets, cs, seen_id) * config['wt_c_l']
        mmd_loss = model.mmd_loss(data, cu) * config['wt_mmd_l']
        loss = loss_flow + centralizing_loss + mmd_loss
        loss.backward()

        nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

        if loss.isnan():
            print('Nan in loss!')
            Exception('Nan in loss!')

        losses.append(loss.item())

        run.log({"loss": loss.item(),
                 "loss_flow": loss_flow.item(),  # }, step=epoch)
                 "loss_central": centralizing_loss.item(),  # }, step=epoch)
                 "loss_mmd": mmd_loss.item(),
                 "ldj": ldj.item(),  # }, step=epoch)
                 "lg": lg.item()})

    if epoch % 2 == 0:
        with torch.no_grad():
            loss_flow_val = 0
            centralizing_loss_val = 0
            mmd_loss_val = 0
            loss_val = 0
            losses_val = []
            for data_val, targets_val, _ in tqdm.tqdm(val_loader, desc=f'Validation Epoch({epoch})'):
                data_val = data_val.to(device)
                targets_val = targets_val.to(device)

                log_prob, _, _ = model.log_prob(data, targets)
                loss_flow_val = - log_prob.mean() * config['wt_f_l']
                centralizing_loss_val = model.centralizing_loss(data_val, targets_val, cs, test_id) * config['wt_c_l']
                mmd_loss_val = model.mmd_loss(data_val, cu) * config['wt_mmd_l']
                loss_val = loss_flow + centralizing_loss + mmd_loss
                losses_val.append(loss_val.item())

        run.log({"loss_val": sum(losses_val) / len(losses_val)})
        run.log({"epoch": epoch})
    print(f'Epoch({epoch}): loss:{sum(losses)/len(losses)}')

    if epoch % 2 == 0:
        state = {'config': config.as_dict(),
                 'split': config['split'],
                 'state_dict': model.state_dict()}

        torch.save(state, f'{SAVE_PATH}{wandb.run.name}-{epoch}.pth')

    if loss.isnan():
        print('Nan in loss!')
        Exception('Nan in loss!')
