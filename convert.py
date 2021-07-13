import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import torchvision.transforms as transforms

class VisualExtractor(object):
    '''Class used in transforms to extract visual features from pretrained networks '''
    def __init__(self, model):
        self.model = timm.create_model(model, pretrained=True, num_classes=0)
        self.model.eval()

        # normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                           std=[0.229, 0.224, 0.225])

        normalize_cub = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                             std=[1.0 / 255, 1.0 / 255, 1.0 / 255])

        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
      #  self.transform.transforms[-1] = normalize_cub

    def __call__(self, sample):
        tensor = self.transform(sample).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        return out.squeeze()
