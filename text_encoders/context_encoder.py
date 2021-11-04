import csv
import re
import os
import pickle
import string

import numpy as np
import scipy.io
import torch
import yaml
from scipy.io import savemat
from tqdm import tqdm

from text_encoders.text_encoder import (AlbertEncoder, BartEncoder,
                                        BertEncoder, BigBirdEncoder,
                                        ProphetNet, SentencePiece,
                                        TFIDF)
from text_encoders.word_embeddings import WordEmbeddings
import nltk
from nltk.corpus import stopwords
# import spacy
# import unidecode
# from word2number import w2n
# import contractions

IDENTITY = '  Context Encoder ~| '

class ContextEncoder():
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device

        if self.config['weighted_encoding']:
            self.tfidf = TFIDF(self.config, device=self.device)

        if 'tfidf' in self.config['text_encoder']:
            self.text_encoder = TFIDF(self.config, device=self.device)
        elif 'prophet' in self.config['text_encoder']:
            self.text_encoder = ProphetNet(self.config, device=self.device)
        elif self.config['text_encoder'] == 'albert':
            self.text_encoder = AlbertEncoder(self.config, device=self.device)
        elif self.config['text_encoder'] == 'bart':
            self.text_encoder = BartEncoder(self.config, device=self.device)
        elif self.config['text_encoder'] == 'bert':
            self.text_encoder = BertEncoder(self.config, device=self.device)
        elif self.config['text_encoder'] == 'bigbird':
            self.text_encoder = BigBirdEncoder(self.config, device=self.device)
        elif 'sentence' in self.config['text_encoder']:
            self.text_encoder = SentencePiece(self.config, device=self.device)
        elif 'glove' in self.config['text_encoder']:
            self.text_encoder = WordEmbeddings(self.config, device=self.device)
        else:
            print(f"{IDENTITY} Encoding setting not found")
            raise Exception

        if self.config['dataset'] == 'imagenet':
            self._encode_contexts_imagenet(weighted_encoding=self.config['weighted_encoding'])
        elif self.config['dataset'] == 'cub2011':
            self._encode_contexts_cub2011(weighted_encoding=self.config['weighted_encoding'])
        else:
            print(f"{IDENTITY} Dataset not found")
            raise Exception

    def _pre_process_articles(self, _articles):
        articles = [re.sub(r'\d+', '', art.lower()).translate(str.maketrans('', '', string.punctuation))
                    for art in _articles]

        nltk.download('stopwords')
        clean_articles = []
        cleaner = tqdm(articles, desc=f'{IDENTITY} Removing all stopwords')
        for art in cleaner:
            clean_articles.append(" ".join([word for word in art.split() if word not in stopwords.words('english')]))
        articles = clean_articles

        return articles

    def _encode_contexts_cub2011(self, weighted_encoding=False):
        wiki_dir = 'data/CUB_200_2011/CUBird_WikiArticles'

        file_list = next(os.walk(wiki_dir), (None, None, []))[2]
        file_list_id = [int(file.split('.')[0]) for file in file_list]
        file_list_ordered = [x for _, x in sorted(zip(file_list_id, file_list))]

        articles = [open(f'{wiki_dir}/{file}').read() for file in file_list_ordered]
        if self.config['preprocess_text']:
            articles = self._pre_process_articles(articles)

        if weighted_encoding:
            _, term_value = self.tfidf(articles)
            self.attributes = np.stack([(self.text_encoder(terms)).mean(axis=0)
                                       for terms, values in term_value]).astype(np.float32)
                # 2 options:
                # - encode all as one
                # - encode each word separatly and weigth it
        else:
            if 'tfidf' in self.config['text_encoder']:
                self.attributes, _ = self.text_encoder(articles)
            else:
                semantic = tqdm(articles, desc=f'{IDENTITY} Encoding All Semantic Descriptions CUB2011')
                with torch.no_grad():
                    contexts = [(self.text_encoder(article)).type(torch.float32) for article in semantic]
                self.attributes = np.stack([feature.cpu().numpy() for feature in contexts])

    def _encode_contexts_imagenet(self, weighted_encoding=False):
        class_ids_dir = "data/image_net/ImageNet-Wiki_dataset/class_article_correspondences/class_article_correspondences_trainval.csv"
        articles_dir = "data/image_net/ImageNet-Wiki_dataset/class_article_text_descriptions/class_article_text_descriptions_trainval.pkl"

        self.wnid_correspondance, articles = self._article_correspondences(class_ids_dir, articles_dir)

        path = 'data/image_net/mat/text/WordEmbeddings_Lo_glove.840B.300d_ImageNet_trainval_classes_classes.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f)

        articles_id = [value['wnid'] for _, value in data.items()]
        articles = [articles[art_id] for art_id in articles_id]

        concat_articles = []
        for article in articles:
            if len(article) == 1:
                concat_articles.append(re.sub('[\n]+', '\n', article[0]))
            else:
                concat_articles.append(re.sub('[\n]+', '\n', '\n'.join(article)))
        articles = concat_articles

        if self.config['preprocess_text']:
            articles = self._pre_process_articles(concat_articles)

        if weighted_encoding:
            _, term_value = self.tfidf(articles)
            term_value_bar = tqdm(articles, desc=f'{IDENTITY} Encoding All Semantic Descriptions ImageNet')
            self.attributes = np.stack([(self.text_encoder(terms)).mean(axis=0)
                                       for terms, values in term_value_bar]).astype(np.float32)
                # 2 options:
                # - encode all as one
                # - encode each word separatly and weigth it
        else:
            if 'tfidf' in self.config['text_encoder']:
                self.attributes, self.term_value_pair = self.text_encoder(articles)
            else:
                semantic = tqdm(articles, desc=f'{IDENTITY} Encoding All Semantic Descriptions ImageNet')
                with torch.no_grad():
                    contexts = [torch.tensor((self.text_encoder(article))).type(torch.float32) for article in semantic]
                self.attributes = np.stack([feature.cpu().numpy() for feature in contexts])

    def _article_correspondences(self, class_article_correspondences_path, class_article_text_descriptions_path):
        articles = pickle.load(
            open(class_article_text_descriptions_path, 'rb')
        )

        temp = [articles[art] for art in articles]
        articles = {t['wnid']: t['articles'] for t in temp}

        with open(class_article_correspondences_path, 'r') as file:
            reader = csv.reader(file)
            article_correspondences = {item[0]: idx for idx, item in enumerate(reader)}  # Make a dictionary out of the csv {wnid: classes}

        return article_correspondences, articles
