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
from dataloaders.cub2011 import Cub2011

import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from convert import CostumTransform
import torchvision.datasets as datasets
import torchvision.transforms as transforms


SAVE_PATH = 'checkpoints/'
os.environ['WANDB_MODE'] = 'offline'
save = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = wandb.init(project='zero_flow_CUB', entity='mvalente',
                 config=r'config/base_conf.yaml')

config = wandb.config

normalize_cub = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0 / 255, 1.0 / 255, 1.0 / 255])
# normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                           std=[0.229, 0.224, 0.225])

transforms_cub = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize_cub,
    transforms.ToPILImage(mode='RGB'),
    CostumTransform(config['image_encoder'])
])

cub = Cub2011(root='/project/data/', transform=transforms_cub, download=False)
seen_id = list(set(cub.data['target']))
unseen_id = list(set(cub.data_unseen['target']))

context_encoder = ContextEncoder(config, seen_id, unseen_id, device)
contexts = context_encoder.contexts.to(device)
cs = context_encoder.cs.to(device)
cu = context_encoder.cu.to(device)


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

generator = Flow(transforms, base_dist)
generator.load_state_dict(torch.load(f'{SAVE_PATH}true-fire-25-15.pth'))
generator = generator.to(device)
generator.eval()

generated_unseen_features = []
number_samples = 60
with torch.no_grad():
    for cu_ in tqdm.tqdm(cu, desc="Generating Unseen Features"):
        generated_unseen_features.append(generator.generation(
            torch.hstack((cu_.repeat(number_samples).reshape(-1, context_dim),
                          visual_distribution.sample([number_samples])))))

cub.insert_generated_features(torch.stack(generated_unseen_features), number_samples)