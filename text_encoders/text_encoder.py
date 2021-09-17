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

class SentencePiece:
    def __init__(self, config, device):
        self.config = config
        self.model = SentenceTransformer('all-mpnet-base-v2', device=device)

    def __call__(self, input_texts):
        sentence_embedding = torch.tensor(self.model.encode(input_texts))
        if isinstance(input_texts, list):
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
        self.device = device
        model_name = self.config['model_name']
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = AlbertModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.summary_extraction_mode = self.config['summary_extraction_mode']

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

        if mode == 'mean_tokens':
            m = utils.reduce_mean_masked(last_hidden_states, mask=attention_mask.unsqueeze(-1), axis=1)
            return m
        elif mode == 'sum_tokens':
            m = utils.reduce_sum_masked(last_hidden_states, mask=attention_mask.unsqueeze(-1), axis=1)
            return m
        elif mode == 'pooler_output':
            return pooler_output
        else:
            # TODO - try other methods
            raise NotImplementedError()
