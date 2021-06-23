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
                 config=r'config/finetune_conf.yaml')

state = torch.load(f'{SAVE_PATH}resilient-snowball-40-12.pth')

generator_config = state['config']

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
    CostumTransform(generator_config['image_encoder'])
])

cub = Cub2011(root='/project/data/', transform=transforms_cub, download=False)
seen_id = list(set(cub.data['target']))
unseen_id = list(set(cub.data_unseen['target']))

context_encoder = ContextEncoder(generator_config, seen_id, unseen_id, device, generation=True)
cu = context_encoder.cu.to(device)


input_dim = cub[0][0].shape.numel()
context_dim = cu[0].shape.numel()
split_dim = input_dim - context_dim

visual_distribution = dist.MultivariateNormal(torch.zeros(split_dim).to(device), torch.eye(split_dim).to(device))

if generator_config['permuter'] == 'random':
    permuter = lambda dim: Permuter(permutation=torch.randperm(dim, dtype=torch.long).to(device))
elif generator_config['permuter'] == 'reverse':
    permuter = lambda dim: Reverse(dim_size=dim)
elif generator_config['permuter'] == 'LinearLU':
    permuter = lambda dim: LinearLU(num_features=dim, eps=1.0e-5)

if generator_config['non_linearity'] == 'relu':
    non_linearity = torch.nn.ReLU()
elif generator_config['non_linearity'] == 'prelu':
    non_linearity = nn.PReLU(init=0.01)
elif generator_config['non_linearity'] == 'leakyrelu':
    non_linearity = nn.LeakyReLU()

if not generator_config['hidden_dims']:
    hidden_dims = [input_dim // 2]
else:
    hidden_dims = generator_config['hidden_dims']

transforms = []
for index in range(generator_config['block_size']):
    if generator_config['act_norm']:
        transforms.append(ActNormBijection(input_dim, data_dep_init=True))
    transforms.append(permuter(input_dim))
    transforms.append(AffineCoupling(
        input_dim,
        split_dim,
        context_dim=context_dim, hidden_dims=hidden_dims, non_linearity=non_linearity, net=generator_config['net']))

generator = Flow(transforms)  # No base distribution needs to be passed since we only want generation
generator.load_state_dict(state['state_dict'])
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

train_loader = torch.utils.data.DataLoader(cub, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

model = timm.create_model(config['image_encoder'], pretrained=True, num_classes=config['num_classes'])
model.train()
model = model.to(device)

print(f'Number of trainable parameters: {sum([x.numel() for x in model.parameters()])}')
run.watch(model)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
loss_fn = nn.CrossEntropyLoss().to(device)

epochs = tqdm.trange(1, config['epochs'])
for epoch in epochs:
    losses = []
    #  for batch_idx, (input, target) in enumerate(loader):
    for data, targets in tqdm.tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        if loss.isnan():
            print('Nan in loss!')
            Exception('Nan in loss!')

        run.log({"loss": loss})
