import ssl

import nltk
import pandas


def download_ntlk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt_tab')
    nltk.download('stopwords')


def load_and_label_data(positive_path, negative_path):
    positive = pandas.read_csv(positive_path, sep=';', usecols=[3], names=['text'])
    negative = pandas.read_csv(negative_path, sep=';', usecols=[3], names=['text'])

    positive['label'] = 'positive'
    negative['label'] = 'negative'

    return pandas.concat([positive, negative])