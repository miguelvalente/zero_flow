import csv
import os
import pickle

import numpy as np
import scipy.io
import torch
import yaml
from scipy.io import savemat
from tqdm import tqdm

from text_encoders.text_encoder import AlbertEncoder, BartEncoder, ProphetNet
from text_encoders.word_embeddings import WordEmbeddings

IDENTITY = 'Context Encoder| '

class ContextEncoder():
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device

        if self.config['load_precomputed_text']:
            self._load_features(self.config['mat_file_text'])
        else:
            if self.config['text_encoder'] == 'prophet_net':
                self.text_encoder = ProphetNet(self.config, device=self.device)
            elif self.config['text_encoder'] == 'albert':
                self.text_encoder = AlbertEncoder(self.config, device=self.device)
            elif self.config['text_encoder'] == 'bart':
                self.text_encoder = BartEncoder(self.config, device=self.device)
            elif self.config['text_encoder'] == 'glove':
                self.text_encoder = WordEmbeddings(self.config, device=self.device)
            else:
                print(f"{IDENTITY} Encoding setting not found")
                raise

            if self.config['dataset'] == 'imagenet':
                self._encode_contexts_imagenet()
            elif self.config['dataset'] == 'cub2011':
                self._encode_contexts_cub2011()
            else:
                print(f"{IDENTITY} Dataset not found")
                raise Exception

    def _load_features(self, mat_file):
        print(f'{IDENTITY} Loading .mat file: {mat_file}')
        if not os.path.exists(mat_file):
            print(f'{IDENTITY} .mat does not exist. Loading not possible using _encode_contexts() instead.')
            self._encode_contexts_cub2011()
        else:
            raw_features = scipy.io.loadmat(mat_file)

            # self.contexts = torch.from_numpy(raw_features['features']).type(torch.float32)
            self.contexts = torch.from_numpy(raw_features['att'].transpose()).type(torch.float32)

    def _encode_contexts_cub2011(self):
        wiki_dir = '/project/data/Raw_Wiki_Articles/CUBird_WikiArticles'

        file_list = next(os.walk(wiki_dir), (None, None, []))[2]
        file_list_id = [int(file.split('.')[0]) for file in file_list]
        file_list_ordered = [x for _, x in sorted(zip(file_list_id, file_list))]

        articles = [open(f'{wiki_dir}/{file}').read() for file in file_list_ordered]

        if self.config['text_encoder'] == 'glove':
            save_path = f"{self.config['save_text_features']}{self.config['glove_dir'].split('/')[-1][:-4]}.mat"
        else:
            save_path = f"{self.config['save_text_features']}{self.config['text_encoder']}.mat"

        if os.path.exists(save_path):
            print(f"{IDENTITY} already used this config to encode text. Using _load_features() instead")
            self._load_features(save_path)
        else:
            semantic = tqdm(articles, desc='Encoding All Semantic Descriptions CUB2011')
            self.contexts = [torch.from_numpy(self.text_encoder.encode_long_text(article)).type(torch.float32) for article in semantic]
            self.contexts = torch.stack(self.contexts)

            if self.config['save_text_features']:
                features = [feature.numpy() for feature in self.contexts]
                mdic = {'features': features}
                savemat(save_path, mdic)

    def _encode_contexts_imagenet(self):
        class_ids_dir = "data/ImageNet-Wiki_dataset/class_article_correspondences/class_article_correspondences_trainval.csv"
        articles_dir = "data/ImageNet-Wiki_dataset/class_article_text_descriptions/class_article_text_descriptions_trainval.pkl"

        a, articles = self._article_correspondences(class_ids_dir, articles_dir)

        image_net_dir = self.config['image_net_dir']

        articles_id = [key for key, value in articles.items()]
        tiny_ids = [dirs for _, dirs, _ in os.walk(image_net_dir)][0]
        articles_id = list(set(articles_id).intersection(tiny_ids))

        articles = [articles[art_id] for art_id in articles_id]

        semantic = tqdm(articles, desc='Encoding Semantic Descriptions')
        contexts = [torch.from_numpy(self.text_encoder.encode_multiple_descriptions(article)) for article in semantic]
        self.contexts = torch.stack(contexts).to(self.device)

    def _article_correspondences(self, class_article_correspondences_path, class_article_text_descriptions_path):
        articles = pickle.load(
            open(class_article_text_descriptions_path, 'rb')
        )

        temp = [articles[art] for art in articles]
        articles = {t['wnid']: t['articles'] for t in temp}

        with open(class_article_correspondences_path, 'r') as file:
            reader = csv.reader(file)
            article_correspondences = {item[0]: item[1:] for item in reader}  # Make a dictionary out of the csv {wnid: classes}

        return article_correspondences, articles
