import torch
import numpy as np
from transformers import (AlbertTokenizer, AlbertModel,
                          ProphetNetTokenizer, ProphetNetEncoder,
                          BartForConditionalGeneration, BartTokenizer,
                          BertModel, BertTokenizer,
                          BigBirdModel, BigBirdTokenizer)
import utils
from text_encoders.text_encoder_utils import split_with_overlap
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from tqdm import tqdm


class TFIDF():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.cv = CountVectorizer()
        self.tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=None, ngram_range=(1, 2))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        # nltk.download('punkt')
        # nltk.download('wordnet')
        # self.tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)

    def test(self, articles):
        cv = CountVectorizer()

        word_count_vector = cv.fit_transform(articles)
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])

        df_idf.sort_values(by=['idf_weights'])

        count_vector = cv.transform(articles)

        tf_idf_vector = tfidf_transformer.transform(count_vector)

        feature_names = cv.get_feature_names()

        for vector in tf_idf_vector:
            df = pd.DataFrame(vector.T.todense(), index=feature_names, columns=["tfidf"])
            print(df.sort_values(by=["tfidf"], ascending=False)[:15], '\n\n')

        print()

    def __call__(self, articles):
        if 'stem' in self.config['text_encoder']:
            articles = [self.stemmer.stem(article) for article in articles]
        if 'lemma' in self.config['text_encoder']:
            articles = [self.lemmatizer.lemmatize(article) for article in articles]

        if 'top_20' in self.config['text_encoder']:
            articles = ['\n'.join(article.split('\n')[:20]) for article in articles]

        vectors = self.tfidf_vectorizer.fit_transform(articles)

        term_value_pair = []
        for v in tqdm(vectors[:10], desc='Getting top 10 terms and corresponding vectors'):
            df = pd.DataFrame(v.T.todense(), index=self.tfidf_vectorizer.get_feature_names(), columns=["tfidf"]) 
            values = list(df.sort_values(by=["tfidf"], ascending=False).values)
            term = list(df.sort_values(by=["tfidf"], ascending=False).index)
            term_value_pair.append((term, values))

        return np.array(vectors.todense()).astype(np.float32), term_value_pair

    def _tokenize(articles):
        preprocessed = []
        preprocessed.append(' '.join([self.stemmer.stem(item) for item in filtered]))
        tokens = nltk.word_tokenize(articles)
        stems = []
        for item in tokens:
            stems.append(PorterStemmer().stem(item))
        return stems

class SentencePiece:
    def __init__(self, config, device):
        self.config = config
        self.model = SentenceTransformer('all-mpnet-base-v2', device=device)

    def __call__(self, input_texts, reduce=False):
        sentence_embedding = torch.tensor(self.model.encode(input_texts))
        if reduce:
            assert isinstance(input_texts, list), "input_text should be 'list' type"
            return sentence_embedding.mean(dim=0)
        else:
            return sentence_embedding

class BigBirdEncoder:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
        self.model = BigBirdModel.from_pretrained('google/bigbird-roberta-base', output_hidden_states=True)
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, input_texts):
        ids = torch.LongTensor(self.tokenizer(input_texts, max_length=4096)['input_ids'])
        ids = ids.to(self.device)

        with torch.no_grad():
            hidden_states = self.model(input_ids=ids.reshape(1, -1))[2]

        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        return sentence_embedding

class BertEncoder:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, input_texts):
        ids = torch.LongTensor(self.tokenizer(input_texts)['input_ids'])
        ids = ids.to(self.device)

        with torch.no_grad():
            hidden_states = self.model(input_ids=ids.reshape(1, -1))[2]

        print()

# Adapted to PyTorch and to my use case from https://github.com/sebastianbujwid/zsl_text_imagenet.git
class BaseEncoder:
    def extract_text_summary(self, last_hidden_states, attention_mask):
        mode = self.summary_extraction_mode

        if mode == 'mean_tokens':
            m = utils.reduce_mean_masked(last_hidden_states, mask=attention_mask.unsqueeze(-1), axis=1)
            return m
        elif mode == 'sum_tokens':
            m = utils.reduce_sum_masked(last_hidden_states, mask=attention_mask.unsqueeze(-1), axis=1)
            return m
        else:
            # TODO - try other methods
            raise NotImplementedError()

    def encode_long_text(self, long_text, batch=32):
        assert isinstance(long_text, str)

        split_text = split_with_overlap(long_text,
                                        max_length=self.config['max_length'],
                                        overlap_window_length=self.config['overlap_window'],
                                        tokenize_func=self.tokenizer.tokenize)  # NOTE: This is not fully correct. Has issues with sub-words
        # (results do not differ much, however).

        encoded_splits = None
        _from = 0
        to = _from + batch
        while _from < len(split_text):
            encoded = self(split_text[_from:to]).detach().cpu().numpy()
            if encoded_splits is None:
                encoded_splits = encoded
            else:
                encoded_splits = np.concatenate([encoded_splits, encoded], axis=0)
            _from = to
            to = _from + batch

        # encoded_splits = self(split_text).numpy()
        return self.aggregate_split_text(encoded_splits)

    def encode_multiple_descriptions(self, long_texts_list):
        assert isinstance(long_texts_list, list)
        agg_method = self.config['aggregate_descriptions_method']

        feats = [self.encode_long_text(t) for t in long_texts_list]
        if agg_method == 'mean_representations':
            return np.mean(feats, axis=0)
        elif agg_method == 'sum_representations':
            return np.sum(feats, axis=0)

        raise ValueError(f'Cannot recognize aggregation method for multiple descriptions: {agg_method}')

    def aggregate_split_text(self, splits):
        method = self.config['aggregate_long_text_splits_method']
        if method == 'mean':
            return np.mean(splits, axis=0)
        elif method == 'sum':
            return np.sum(splits, axis=0)
        else:
            # TODO - try other methods
            raise NotImplementedError()

class BartEncoder(BaseEncoder):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        self.model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
        self.model = self.model.to(device)
        self.summary_extraction_mode = self.config['summary_extraction_mode']

    def __call__(self, input_texts, **kwargs):
        inputs = self.tokenizer(input_texts,
                                return_tensors='pt',
                                padding=True).to(self.device)

        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        return self.extract_text_summary(outputs['encoder_last_hidden_state'], inputs['attention_mask'])

class ProphetNet(BaseEncoder):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
        self.model = ProphetNetEncoder.from_pretrained('patrickvonplaten/prophetnet-large-uncased-standalone')
        self.model = self.model.to(device)
        self.summary_extraction_mode = self.config['summary_extraction_mode']

    def __call__(self, input_texts, **kwargs):
        inputs = self.tokenizer.batch_encode_plus(input_texts,
                                                  add_special_tokens=True,
                                                  return_tensors='pt',
                                                  padding=True).to(self.device)

        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        return self.extract_text_summary(outputs['last_hidden_state'], inputs['attention_mask'])


class AlbertEncoder(BaseEncoder):
    def __init__(self, config, device):
        self.config = config
        self.device = 'cpu'
        model_name = 'albert-base-v2'
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = AlbertModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.summary_extraction_mode = 'mean_tokens'

    def __call__(self, input_texts, **kwargs):
        inputs = self.tokenizer.batch_encode_plus(input_texts, add_special_tokens=True, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids']
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return self.extract_text_summary(outputs, attention_mask)

    def extract_text_summary(self, outputs, attention_mask):
        mode = self.summary_extraction_mode
        pooler_output = outputs['pooler_output']
        last_hidden_states = outputs['last_hidden_state']

        m = utils.reduce_mean_masked(last_hidden_states, mask=attention_mask.unsqueeze(-1), axis=1)

        return m.mean(axis=0)

    def encode_long_text(self, long_text, batch=32):
        assert isinstance(long_text, str)

        split_text = split_with_overlap(long_text,
                                        max_length=256,
                                        overlap_window_length=50,
                                        tokenize_func=self.tokenizer.tokenize)  # NOTE: This is not fully correct. Has issues with sub-words (results do not differ much, however).

        encoded_splits = None
        _from = 0
        to = _from + batch
        while _from < len(split_text):
            encoded = self(split_text[_from:to]).numpy()
            if encoded_splits is None:
                encoded_splits = encoded
            else:
                encoded_splits = np.concatenate([encoded_splits, encoded], axis=0)
            _from = to
            to = _from + batch

        return self.aggregate_split_text(encoded_splits)

    def encode_multiple_descriptions(self, long_texts_list):
        assert isinstance(long_texts_list, list)

        feats = [self.encode_long_text(t) for t in long_texts_list]
        return np.mean(feats, axis=0)

    def aggregate_split_text(self, splits):
        return np.mean(splits, axis=0)
