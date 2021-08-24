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
from dataloaders.cub2011 import Cub2011
from distributions import DoubleDistribution, SemanticDistribution
from nets import Classifier
from permuters import LinearLU, Permuter, Reverse
from text_encoders.context_encoder import ContextEncoder
from transform import Flow

# CUDA_LAUNCH_BLOCKING = 1
SAVE_PATH = 'checkpoints/'
os.environ['WANDB_MODE'] = 'online'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = wandb.init(project='zero_classifier_CUB', entity='mvalente',
                 config=r'config/classifier.yaml')

wandb.config['checkpoint'] = 'major-dawn-57-20.pth'

state = torch.load(f"{SAVE_PATH}{wandb.config['checkpoint']}")
wandb.config['split'] = state['split']

wandb.config.update(state['config'])
with open('config/dataloader.yaml', 'r') as d, open('config/context_encoder.yaml', 'r') as c:
    wandb.config.update(yaml.safe_load(d))
    wandb.config.update(yaml.safe_load(c), allow_val_change=True)

config = wandb.config

transforms_cub = None
# if config['image_encoder']:
#     transforms_cub = transforms.Compose([
#         VisualExtractor(config['image_encoder'])
#     ])

cub = Cub2011(root='/project/data/', zero_shot=True, config=config, transform=transforms_cub)
generation_ids = cub.generation_ids
imgs_per_class = cub.imgs_per_class

context_encoder = ContextEncoder(config, device=device)
contexts = context_encoder.contexts.to(device)[generation_ids]

input_dim = 2048  # cub[0][0].shape.numel()
context_dim = contexts[0].shape.numel()
split_dim = input_dim - context_dim

semantic_distribution = SemanticDistribution(contexts, torch.ones(context_dim).to(device))
visual_distribution = dist.MultivariateNormal(torch.zeros(split_dim).to(device), torch.eye(split_dim).to(device))

if config['permuter'] == 'random':
    permuter = lambda dim: Permuter(permutation=torch.randperm(dim, dtype=torch.long).to(device))
elif config['permuter'] == 'reverse':
    permuter = lambda dim: Reverse(dim_size=dim)
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

generator = Flow(transform)  # No base distribution needs to be passed since we only want generation
generator.load_state_dict(state['state_dict'])
generator = generator.to(device)
generator.eval()

generated_features = []
labels = []
with torch.no_grad():
    for idx, class_id in enumerate(tqdm.tqdm(generation_ids, desc='Generating Features')):
        num_samples = imgs_per_class[class_id]
        if config['context_sampling']:
            s = semantic_distribution.sample(num_samples=num_samples, n_points=1, context=idx).reshape(num_samples, -1)
            v = visual_distribution.sample([num_samples])
            prime_generation = torch.cat((s, v), axis=1)
            generated_features.append(generator.generation(prime_generation))
        else:
            generated_features.append(generator.generation(
                torch.hstack((contexts[idx].repeat(num_samples).reshape(-1, context_dim),
                              visual_distribution.sample([num_samples])))))

generated_features = torch.cat(generated_features, dim=0).to('cpu')

labels = list(np.concatenate([np.repeat(idx, imgs_per_class[class_id]) for idx, class_id in enumerate(generation_ids)]))
cub.insert_generated_features(generated_features, labels)

try:  # Deletes genererator model since its not used anymore
    del generator
    torch.cuda.empty_cache()
except Exception:
    print("Failed to delete generator or clear CUDA cache memory")

train_loader = torch.utils.data.DataLoader(cub, batch_size=config['batch_size_c'], shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(cub, batch_size=1000, shuffle=True, pin_memory=True)

model = Classifier(input_dim, len(generation_ids))  # len of gen_ids corresponds to the number of classes to classify
model.train()
model = model.to(device)


print(f'Number of trainable parameters: {sum([x.numel() for x in model.parameters()])}')
run.watch(model)

if config['tau']:
    loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)
    tau = config['tau']
    tau = torch.tensor(tau).to(device)
else:
    loss_fn = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=config['lr_c'])
for epoch in range(config['epochs_c']):
    losses = []
    for data, targets, seen_or_unseen in tqdm.tqdm(train_loader, desc=f'Epoch({epoch})'):
        data = data.to(device)
        targets = targets.to(device)
        seen_or_unseen = seen_or_unseen.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, targets)

        if config['tau']:
            cal_stacking = seen_or_unseen + 1.0
            cal_stacking[cal_stacking == 2] = tau
            loss = (loss * cal_stacking).mean()

        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        wandb.log({"loss": loss.item()})

    if True:
        with torch.no_grad():
            correct_seen = 0
            total_unseen = 0
            accuracy_unseen = 0
            correct_unseen = 0
            total_seen = 0
            accuracy_seen = 0
            correct_test = 0
            total_test = 0
            accuracy_test = 0
            harmonic_mean = 0

            cub.eval()  # Switch dataset return to img, target, seen_or_unseen
            for data_val, target_val, seen_or_unseen in tqdm.tqdm(val_loader, desc="Validation"):
                images = data_val.to(device)
                labels = target_val.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                if 'seen' in config['zero_test'] or 'all' in config['zero_test']:
                    total_seen += seen_or_unseen[seen_or_unseen == 1].numel()
                    correct_seen += (predicted[seen_or_unseen == 1] == labels[seen_or_unseen == 1]).sum().item()
                if 'zero' in config['zero_test'] or 'all' in config['zero_test']:
                    total_unseen += seen_or_unseen[seen_or_unseen == 0].numel()
                    correct_unseen += (predicted[seen_or_unseen == 0] == labels[seen_or_unseen == 0]).sum().item()
                if 'all' in config['zero_test']:
                    total_test += seen_or_unseen[seen_or_unseen == 2].numel()
                    correct_test += (predicted[seen_or_unseen == 2] == labels[seen_or_unseen == 2]).sum().item()

            # print(f'correct seen:{correct_seen}  correct unseen:{correct_unseen} | total s:{total_seen}  total u:{total_unseen}')

            if 'seen' in config['zero_test'] or 'all' in config['zero_test']:
                if correct_seen != 0:
                    accuracy_seen = correct_seen / total_seen
                wandb.log({"Acc_seen": accuracy_seen})

            if 'zero' in config['zero_test'] or 'all' in config['zero_test']:
                if correct_unseen != 0:
                    accuracy_unseen = correct_unseen / total_unseen
                wandb.log({"Acc_unseen": accuracy_unseen})

            if 'seen' in config['zero_test'] and 'zero' in config['zero_test']:
                if accuracy_seen != 0 and accuracy_unseen != 0:
                    harmonic_mean = 2 / (1 / accuracy_seen + 1 / accuracy_unseen)
                wandb.log({"Harmonic Mean": harmonic_mean})

            if 'all' in config['zero_test']:
                if correct_test != 0:
                    accuracy_test = correct_test / total_test
                    wandb.log({"Acc_test": accuracy_test})
            cub.eval()  # Switch dataset return to img, target

    print(f'real: {len(cub.test_real)} | gen: {len(cub.test_gen)}')

    if loss.isnan():
        raise Exception('Nan in loss!')
