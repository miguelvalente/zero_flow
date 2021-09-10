import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import torchvision.transforms as transforms

class VisualExtractor():
    '''Class used in transforms to extract visual features from pretrained networks '''
    def __init__(self, config):
        model = config['image_encoder']
        self.model = timm.create_model(model, pretrained=True, num_classes=0)
        self.model.eval()

        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        self.transform.transforms = self.transform.transforms[:-1]

        print(f'\n Visual Encoder Summary: \n Model: {model} \n Transforms: {self.transform}')

    def __call__(self, sample):
        tensor = self.transform(sample).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        return out.squeeze()
