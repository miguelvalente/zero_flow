from text_encoders.text_encoder import AlbertEncoder
import numpy as np
from data_utils import article_correspondences

path = "data/class_article_correspondences/class_article_correspondences_mp500.csv"
path2 = "data/class_article_text_descriptions/class_article_text_descriptions_mp500.pkl"

article_correspondences, articles = article_correspondences(path, path2)

albert = AlbertEncoder({
    'model_name': 'albert-base-v2',
    'summary_extraction_mode': 'sum_tokens',
    'aggregate_long_text_splits_method': 'mean',
    'aggregate_descriptions_method': 'sum_representations',
    'overlap_window': 5,
    'max_length': 20, })

texts = ['Very short',
         'A long text. ' * 20,
         'Even longer text than the previous. ' * 100]
batch_1_emb = []
for t in texts:
    t_emb = albert.encode_multiple_descriptions([t])
    batch_1_emb.append(t_emb)

batch_1_emb = np.sum(batch_1_emb, axis=0)

batch_2_emb = albert.encode_multiple_descriptions(articles[1007]['articles'])

np.testing.assert_almost_equal(batch_2_emb, batch_1_emb)
