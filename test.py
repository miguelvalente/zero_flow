import os

from text_encoders.text_encoder import AlbertEncoder
import numpy as np
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import wandb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import yaml


SAVE_PATH = 'checkpoints/'
os.environ['WANDB_MODE'] = 'offline'
save = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = wandb.init(project='zero_flow_CUB', entity='mvalente',
                 config=r'config/base_conf.yaml')

with open('config/base_conf.yaml') as f:
    ya = yaml.load(f, Loader=yaml.FullLoader)
config = wandb.config


state = {'config': config.as_dict(),
         'state_dict': torch.ones(2)}

torch.save(state, 'teste_state.pt')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# valdir = '/project/data/val'

# val_dataset = datasets.ImageFolder(
#     valdir,
#     transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))


# print()


model = timm.create_model('tf_efficientnet_l2_ns', pretrained=True, num_classes=0)
model.eval()

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

img = Image.open('/project/data/train/n01440764/n01440764_18.JPEG').convert('RGB')
tensor = transform(img).unsqueeze(0)  # transform and add batch dimension


with torch.no_grad():
    out = model(tensor)
    print(out.shape)


# # Get imagenet class mappings
# url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
# urllib.request.urlretrieve(url, filename) 
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]

# # Print top categories per image
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# for i in range(top5_prob.size(0)):
#     print(categories[top5_catid[i]], top5_prob[i].item())

# print()
# prints class names and probabilities like:
# [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]


# albert = AlbertEncoder({
#     'model_name': 'albert-base-v2',
#     'summary_extraction_mode': 'sum_tokens',
#     'aggregate_long_text_splits_method': 'mean',
#     'aggregate_descriptions_method': 'sum_representations',
#     'overlap_window': 5,
#     'max_length': 20, })

# texts = ['Very short',
#          'A long text. ' * 20,
#          'Even longer text than the previous. ' * 100]
# batch_1_emb = []
# for t in texts:
#     t_emb = albert.encode_multiple_descriptions([t])
#     batch_1_emb.append(t_emb)

# batch_1_emb = np.sum(batch_1_emb, axis=0)

# batch_2_emb = albert.encode_multiple_descriptions(texts)

# np.testing.assert_almost_equal(batch_2_emb, batch_1_emb)
