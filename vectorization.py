from string import punctuation

from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from data_loading import download_ntlk_data


def vectorize_texts(x_train, x_test, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(ngram_range=ngram_range)

    vectorized_x_train = vectorizer.fit_transform(x_train)

    vectorized_x_test = vectorizer.transform(x_test)

    return vectorized_x_train, vectorized_x_test

def t_fidf_vectorize_texts(x_train, x_test, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    vectorized_x_train = vectorizer.fit_transform(x_train)

    vectorized_x_test = vectorizer.transform(x_test)

    return vectorized_x_train, vectorized_x_test


def vectorize_texts_with_tokens_and_stopwords(x_train, x_test, ngram_range=(1, 1)):
    download_ntlk_data()
    noise = stopwords.words('russian') + list(punctuation)
    vectorizer = CountVectorizer(ngram_range=ngram_range, tokenizer=word_tokenize, stop_words=noise)
    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)

    return vectorized_x_train, vectorized_x_test


def vectorize_texts_without_stopwords(x_train, x_test, vectorizer_class=CountVectorizer, ngram_range=(1, 1)):
    download_ntlk_data()
    vectorizer = vectorizer_class(ngram_range=ngram_range, tokenizer=word_tokenize)

    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)

    return vectorized_x_train, vectorized_x_test


def vectorize_char(x_train, x_test, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)

    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)

    return vectorized_x_train, vectorized_x_test
