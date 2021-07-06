import csv
import os

import numpy as np
import pickle
import torch
from tqdm import tqdm
from text_encoders.text_encoder import AlbertEncoder, ProphetNet, BartEncoder
import yaml

class ContextEncoder():
    def __init__(self, config, seen_id, unseen_id, device, generation=False):
        self.config = config
        self.device = device
        self.seen_id = np.array(seen_id)
        self.unseen_id = np.array(unseen_id)
        self.generation = generation

        if self.config['text_encoder'] == 'prophet_net':
            self.text_encoder = ProphetNet(self.config, device=self.device)
        elif self.config['text_encoder'] == 'albert':
            self.text_encoder = AlbertEncoder(self.config, device=self.device)
        elif self.config['text_encoder'] == 'bart':
            self.text_encoder = BartEncoder(self.config, device=self.device)
        else:
            print("Model not found")
            raise

        if self.config['dataset'] == 'imagenet':
            self.encode_contexts_imagenet()
        elif self.config['dataset'] == 'cub2011':
            self.encode_contexts_cub2011()
        else:
            print("Dataset not found")
            raise

    def encode_contexts_cub2011(self):
        dira = '/project/data/Raw_Wiki_Articles/CUBird_WikiArticles'

        file_list = next(os.walk(dira), (None, None, []))[2]
        file_list_id = [int(file.split('.')[0]) for file in file_list]
        file_list_ordered = [x for _, x in sorted(zip(file_list_id, file_list))]

        articles = [open(f'{dira}/{file}').read() for file in file_list_ordered]

        if self.generation:
            articles = [articles[i] for i in self.unseen_id]
            semantic = tqdm(articles, desc='Encoding Unseen Classes Semantic Descriptions CUB2011')

            self.cu = [torch.from_numpy(self.text_encoder.encode_long_text(article)) for article in semantic]
            self.cu = torch.stack(self.cu)
        else:
            semantic = tqdm(articles, desc='Encoding All Semantic Descriptions CUB2011')

            self.contexts = [torch.from_numpy(self.text_encoder.encode_long_text(article)) for article in semantic]

            self.contexts = torch.stack(self.contexts)
            self.cs = self.contexts[self.seen_id]
            self.cu = self.contexts[self.unseen_id]

    def encode_contexts_imagenet(self):
        class_ids_dir = "data/ImageNet-Wiki_dataset/class_article_correspondences/class_article_correspondences_trainval.csv"
        articles_dir = "data/ImageNet-Wiki_dataset/class_article_text_descriptions/class_article_text_descriptions_trainval.pkl"

        a, articles = self.article_correspondences(class_ids_dir, articles_dir)

        image_net_dir = self.config['image_net_dir']

        articles_id = [key for key, value in articles.items()]
        tiny_ids = [dirs for _, dirs, _ in os.walk(image_net_dir)][0]
        articles_id = list(set(articles_id).intersection(tiny_ids))

        articles = [articles[art_id] for art_id in articles_id]

        semantic = tqdm(articles, desc='Encoding Semantic Descriptions')
        contexts = [torch.from_numpy(self.text_encoder.encode_multiple_descriptions(article)) for article in semantic]
        self.contexts = torch.stack(contexts).to(self.device)

    def article_correspondences(self, class_article_correspondences_path, class_article_text_descriptions_path):
        articles = pickle.load(
            open(class_article_text_descriptions_path, 'rb')
        )

        temp = [articles[art] for art in articles]
        articles = {t['wnid']: t['articles'] for t in temp}

        with open(class_article_correspondences_path, 'r') as file:
            reader = csv.reader(file)
            article_correspondences = {item[0]: item[1:] for item in reader}  # Make a dictionary out of the csv {wnid: classes}

        return article_correspondences, articles


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("/project/config/base_conf.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile)
