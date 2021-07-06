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
from text_encoders.text_encoder import ProphetNet, AlbertEncoder
from text_encoders.context_encoder import ContextEncoder
from dataloaders.cub2011 import Cub2011

import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from convert import VisualExtractor
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# CUDA_LAUNCH_BLOCKING = 1
SAVE_PATH = 'checkpoints/'
os.environ['WANDB_MODE'] = 'offline'
save = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = wandb.init(project='zero_flow_CUB', entity='mvalente',
                 config=r'config/flow_conf.yaml')

config = wandb.config

transforms_cub = transforms.Compose([
    VisualExtractor(config['image_encoder'])
])

cub_train = Cub2011(which_split='train', root='/project/data/', split=config['split'], transform=transforms_cub, download=False)
seen_id = cub_train.seen_id
unseen_id = cub_train.unseen_id

context_encoder = ContextEncoder(config, seen_id, unseen_id, device)
contexts = context_encoder.contexts.to(device)
cs = context_encoder.cs.to(device)
cu = context_encoder.cu.to(device)

train_loader = torch.utils.data.DataLoader(cub_train, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

# cub_val = Cub2011(which_split='val', root='/project/data/', split=config['split'], transform=transforms_cub, download=False)
# val_loader = torch.utils.data.DataLoader(cub_val, batch_size=1000, shuffle=True, pin_memory=True)

input_dim = cub_train[0][0].shape.numel()
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
        context_dim=context_dim, hidden_dims=hidden_dims, non_linearity=non_linearity, net=config['net']))

model = Flow(transforms, base_dist)
model.train()
model = model.to(device)

print(f'Number of trainable parameters: {sum([x.numel() for x in model.parameters()])}')
run.watch(model)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

for epoch in range(1, config['epochs']):
    losses = []
    for data, targets in tqdm.tqdm(train_loader, desc=f'Epoch({epoch})'):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss_flow = - model.log_prob(data, targets).mean() * config['wt_f_l']
        centralizing_loss = model.centralizing_loss(data, targets, cs, seen_id) * config['wt_c_l']
        mmd_loss = model.mmd_loss(data, cu) * config['wt_mmd_l']
        loss = loss_flow + centralizing_loss + mmd_loss
        loss.backward()
        optimizer.step()

        if loss.isnan():
            print('Nan in loss!')
            Exception('Nan in loss!')

        losses.append(loss.item())

        run.log({"loss": loss.item(),
                 "loss_flow": loss_flow.item(),  # }, step=epoch)
                 "loss_central": centralizing_loss.item(),  # }, step=epoch)
                 "loss_mmd": mmd_loss.item()})

    with torch.no_grad():
        for data_val, targets_val in tqdm.tqdm(val_loader, desc=f'Validation Epoch({epoch})'):
            data_val = data_val.to(device)
            targets_val = targets_val.to(device)

            loss_flow_val = - model.log_prob(data_val, targets_val).mean() * config['wt_f_l']
            centralizing_loss_val = model.centralizing_loss(data_val, targets_val, cs, seen_id) * config['wt_c_l']
            mmd_loss_val = model.mmd_loss(data_val, cu) * config['wt_mmd_l']
            loss_val = loss_flow + centralizing_loss + mmd_loss

            run.log({"loss_val": loss_val.item()})

    run.log({"epoch": epoch})
    print(f'Epoch({epoch}): loss:{sum(losses)/len(losses)}')

    if epoch % 20 == 0:
        state = {'config': config.as_dict(),
                 'split': config['split'],
                 'state_dict': model.state_dict()}

        torch.save(state, f'{SAVE_PATH}{wandb.run.name}-{epoch}.pth')

    if loss.isnan():
        print('Nan in loss!')
        Exception('Nan in loss!')
