import csv
import os
import pickle

import numpy as np
import scipy.io
import torch
import yaml
from tqdm import tqdm

from text_encoders.text_encoder import AlbertEncoder, BartEncoder, ProphetNet
from text_encoders.word_embeddings import WordEmbeddings


class ContextEncoder():
    def __init__(self, config, generation_ids=None, seen_id=None, unseen_id=None, device='cpu', generation=False):
        self.config = config
        self.device = device
        self.seen_id = np.array(seen_id)
        self.unseen_id = np.array(unseen_id)
        self.generation_ids = np.array(generation_ids)
        self.generation = generation

        if self.config['text_encoder'] == 'prophet_net':
            self.text_encoder = ProphetNet(self.config, device=self.device)
        elif self.config['text_encoder'] == 'albert':
            self.text_encoder = AlbertEncoder(self.config, device=self.device)
        elif self.config['text_encoder'] == 'bart':
            self.text_encoder = BartEncoder(self.config, device=self.device)
        elif self.config['text_encoder'] == 'glove':
            self.text_encoder = WordEmbeddings(self.config, device=self.device)
        elif self.config['text_encoder'] == 'manual':
            print('Using manual attributes')
        else:
            print("Model not found")
            raise

        if self.config['dataset'] == 'imagenet':
            self.encode_contexts_imagenet()
        elif self.config['dataset'] == 'cub2011':
            self.encode_contexts_cub2011()
        elif self.config['dataset'] == 'manual_cub2011':
            self.load_contexts_cub2011()
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
            articles = [articles[i] for i in self.generation_ids]
            semantic = tqdm(articles, desc='Generation: Encoding Classes Semantic Descriptions CUB2011')

            self.contexts = [torch.from_numpy(self.text_encoder.encode_long_text(article)).type(torch.float32) for article in semantic]
            self.contexts = torch.stack(self.contexts)
            # self.contexts = torch.ones((200, 1024))
        else:
            semantic = tqdm(articles, desc='Encoding All Semantic Descriptions CUB2011')

            self.contexts = [torch.from_numpy(self.text_encoder.encode_long_text(article)).type(torch.float32) for article in semantic]
            # self.contexts = torch.ones((200, 1024))

            self.contexts = torch.stack(self.contexts)

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

    def load_contexts_cub2011(self):
        raw_att = scipy.io.loadmat('data/xlsa17/data/CUB/att_splits.mat')

        self.contexts = torch.from_numpy(raw_att['att'].transpose()).type(torch.float32)

        if self.generation:
            self.contexts = self.contexts[self.generation_ids]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("/project/config/base_conf.yaml", "r") as ymlfile:
        config = yaml.load(ymlfile)
