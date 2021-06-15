from multiprocessing import Value
import os

import torch
from tqdm import tqdm
from text_encoders.text_encoder import AlbertEncoder, ProphetNet
from utils import article_correspondences
import yaml

class Context():
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device

        self.encode_contexts()

    def __call__(self):
        return self.contexts

    def encode_contexts(self):

        class_ids_dir = "data/ImageNet-Wiki_dataset/class_article_correspondences/class_article_correspondences_trainval.csv"
        articles_dir = "data/ImageNet-Wiki_dataset/class_article_text_descriptions/class_article_text_descriptions_trainval.pkl"

        a, articles = article_correspondences(class_ids_dir, articles_dir)

        if self.config['text_encoder'] == 'prophet_net':
            text_encoder = ProphetNet(self.config, device=self.device)
        elif self.config['text_encoder'] == 'albert':
            text_encoder = AlbertEncoder(self.config, device=self.device)
        else:
            print("Model not found")
            raise

        image_net_dir = self.config['image_net_dir']

        articles_id = [key for key, value in articles.items()]
        tiny_ids = [dirs for _, dirs, _ in os.walk(image_net_dir)][0]
        articles_id = list(set(articles_id).intersection(tiny_ids))

        articles = [articles[art_id] for art_id in articles_id]

        semantic = tqdm(articles, desc='Encoding Semantic Descriptions')
        contexts = [torch.from_numpy(text_encoder.encode_multiple_descriptions(article)) for article in semantic]
        self.contexts = torch.stack(contexts).to(self.device)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("/project/config/base_conf.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile)

    context = Context(config, device)
