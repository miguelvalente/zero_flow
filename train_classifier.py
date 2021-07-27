import os

import numpy as np
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
from dataloaders.cub2011 import Cub2011Zero, Cub2011Zero_Pre
from distributions import DoubleDistribution, SemanticDistribution
from nets import Classifier
from permuters import LinearLU, Permuter, Reverse
from text_encoders.context_encoder import ContextEncoder
from transform import Flow

# CUDA_LAUNCH_BLOCKING = 1
SAVE_PATH = 'checkpoints/'
os.environ['WANDB_MODE'] = 'offline'
# os.environ['WANDB_NAME'] = 'INN'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save = False

run = wandb.init(project='zero_inference_CUB', entity='mvalente',
                 config=r'config/finetune_conf.yaml')

wandb.config['checkpoint'] = 'dazzling-sound-1-340.pth'

state = torch.load(f"{SAVE_PATH}{wandb.config['checkpoint']}")
wandb.config['split'] = state['split']

config = wandb.config
generator_config = state['config']
wandb.config['text_encoder'] = generator_config['text_encoder']
wandb.config['image_encoder'] = generator_config['image_encoder']

normalize_cub = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0 / 255, 1.0 / 255, 1.0 / 255])
# normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                           std=[0.229, 0.224, 0.225])

transforms_cub = transforms.Compose([
    VisualExtractor(generator_config['image_encoder'])
])

cub = Cub2011Zero_Pre(root='/project/data/', test=config['test'], split=config['split'], config=config, transform=transforms_cub)
generation_ids = cub.generation_ids
imgs_per_class = cub.imgs_per_class

context_encoder = ContextEncoder(generator_config, generation_ids=generation_ids, device=device, generation=True)
contexts = context_encoder.contexts.to(device)

input_dim = 2048  # cub[0][0].shape.numel()
context_dim = contexts[0].shape.numel()
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

generated_features = []
labels = []
with torch.no_grad():
    for idx, class_id in enumerate(tqdm.tqdm(generation_ids, desc='Generating Features')):
        number_samples = imgs_per_class[class_id]

        generated_features.append(generator.generation(
            torch.hstack((contexts[idx].repeat(number_samples).reshape(-1, context_dim),
                          visual_distribution.sample([number_samples])))))

generated_features = torch.cat(generated_features, dim=0).to('cpu')

labels = list(np.concatenate([np.repeat(idx, imgs_per_class[class_id]) for idx, class_id in enumerate(generation_ids)]))
cub.insert_generated_features(generated_features, labels)

try:  # Deletes genererator model since its not used anymore
    del generator
    torch.cuda.empty_cache()
except Exception:
    print("Failed to delete generator or clear CUDA cache memory")

train_loader = torch.utils.data.DataLoader(cub, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(cub, batch_size=1000, shuffle=True, pin_memory=True)

model = Classifier(input_dim, config['num_classes'])
model.train()
model = model.to(device)

print(f'Number of trainable parameters: {sum([x.numel() for x in model.parameters()])}')
run.watch(model)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
tau = config['tau']
tau = torch.tensor(tau).to(device)
for epoch in range(config['epochs']):
    losses = []
    #  for batch_idx, (input, target) in enumerate(loader):
    for data, targets, seen_or_unseen in tqdm.tqdm(train_loader, desc=f'Epoch({epoch})'):
        data = data.to(device)
        targets = targets.to(device)
        seen_or_unseen = seen_or_unseen.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, targets)

        # cal_stacking = seen_or_unseen + 1.0
        # cal_stacking[cal_stacking == 2] = tau
        # loss = (loss * cal_stacking).mean()

        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        wandb.log({"loss": loss.item()})

    if True:
        with torch.no_grad():
            correct_seen = 0
            correct_unseen = 0
            correct_test = 0
            total_seen = 0
            total_unseen = 0
            total_test = 0
            accuracy_seen = 0
            accuracy_unseen = 0
            accuracy_test = 0
            harmonic_mean = 0

            cub.eval()  # Switch dataset return to img, target, seen_or_unseen
            for data_val, target_val, seen_or_unseen in tqdm.tqdm(val_loader, desc="Validation"):
                images = data_val.to(device)
                labels = target_val.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total_seen += seen_or_unseen[seen_or_unseen == 1].numel()
                total_unseen += seen_or_unseen[seen_or_unseen == 0].numel()
                total_test += seen_or_unseen[seen_or_unseen == 2].numel()

                correct_seen += (predicted[seen_or_unseen == 1] == labels[seen_or_unseen == 1]).sum().item()
                correct_unseen += (predicted[seen_or_unseen == 0] == labels[seen_or_unseen == 0]).sum().item()
                correct_test += (predicted[seen_or_unseen == 2] == labels[seen_or_unseen == 2]).sum().item()

            print(f'correct seen:{correct_seen}  correct unseen:{correct_unseen} | total s:{total_seen}  total u:{total_unseen}')

            accuracy_seen = correct_seen / total_seen
            accuracy_unseen = correct_unseen / total_unseen
            if accuracy_seen != 0 and accuracy_unseen != 0:
                harmonic_mean = 2 / (1 / accuracy_seen +
                                     1 / accuracy_unseen)

            if correct_test != 0:
                accuracy_test = correct_test / total_test

            wandb.log({"Acc_seen": accuracy_seen,
                       "Acc_unseen": accuracy_unseen,
                       "Harmonic Mean": harmonic_mean,
                       "Epoch": epoch})

            if config['test']:
                wandb.log({"Acc_test": accuracy_test})

            cub.eval()  # Switch dataset return to img, target
    print(f'real: {len(cub.test_real)} | gen: {len(cub.test_gen)}')
    if loss.isnan():
        print('Nan in loss!')
        raise Exception('Nan in loss!')
