import pickle
import csv


def article_correspondences(class_article_correspondences_path, class_article_text_descriptions_path):
    articles = pickle.load(
        open(class_article_text_descriptions_path, 'rb')
    )

    with open(class_article_correspondences_path, 'r') as file:
        reader = csv.reader(file)
        article_correspondences = {item[0]: item[1:] for item in reader}  # Make a dictionary out of the csv {wnid: classes}

    return article_correspondences, articles
