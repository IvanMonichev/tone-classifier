import ssl
from string import punctuation

import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


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
    positive = pd.read_csv(positive_path, sep=';', usecols=[3], names=['text'])
    negative = pd.read_csv(negative_path, sep=';', usecols=[3], names=['text'])

    positive['label'] = 'positive'
    negative['label'] = 'negative'

    return pd.concat([positive, negative])


def vectorize_texts(x_train, x_test, vectorizer_class=CountVectorizer, ngram_range=(1, 1)):
    vectorizer = vectorizer_class(ngram_range=ngram_range)

    vectorized_x_train = vectorizer.fit_transform(x_train)

    vectorized_x_test = vectorizer.transform(x_test)

    return vectorized_x_train, vectorized_x_test, vectorizer


def vectorize_texts_with_tokens(x_train, x_test, vectorizer_class=CountVectorizer, ngram_range=(1, 1)):
    noise = stopwords.words('russian') + list(punctuation)
    print(noise)
    vectorizer = vectorizer_class(ngram_range=ngram_range, tokenizer=word_tokenize, stop_words=noise)
    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)

    return vectorized_x_train, vectorized_x_test


def vectorize_texts_no_stopwords(x_train, x_test, vectorizer_class=CountVectorizer, ngram_range=(1, 2)):
    vectorizer = vectorizer_class(ngram_range=ngram_range, tokenizer=word_tokenize)

    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)

    return vectorized_x_train, vectorized_x_test


def classify_with_char_vectorizer(x_train, x_test):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 1))

    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)

    return vectorized_x_train, vectorized_x_test


def train_logistic_regression(x_train, y_train):
    clf = LogisticRegression(random_state=24, max_iter=1000)
    clf.fit(x_train, y_train)
    return clf


def train_xgb_classifier(x_train, y_train):
    xgb_clf = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=3,
        gamma=0.2,
        subsample=0.6,
        colsample_bytree=1.0,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27
    )

    xgb_clf.fit(x_train, y_train)
    return xgb_clf


def evaluate_model(clf, x_test, y_test):
    pred = clf.predict(x_test)
    print(classification_report(y_test, pred))


if __name__ == '__main__':
    download_ntlk_data()

    # Шаг 1: Загрузка данных и присвоение меток
    dataframe = load_and_label_data('./content/positive.csv', './content/negative.csv')

    # Шаг 2: Разделение данных на тренировочные и тестовые выборки
    x_train, x_test, y_train, y_test = train_test_split(dataframe['text'], dataframe['label'], random_state=24)

    # Шаг 3: Векторизация
    # vectorized_x_train, vectorized_x_test, vectorizer = vectorize_texts(x_train, x_test, CountVectorizer, (1, 1))

    # # TF-IDF векторизация
    # vectorized_x_train, vectorized_x_test, vectorizer = vectorize_texts_with_tokens(x_train, x_test, TfidfVectorizer, (1, 1))

    # Умная векторизация с токенами
    # vectorized_x_train, vectorized_x_test = vectorize_texts_with_tokens(x_train, x_test, CountVectorizer,(1, 2))

    # Векторизация без стоп-слов
    # vectorized_x_train, vectorized_x_test = vectorize_texts_no_stopwords(x_train, x_test, CountVectorizer, (1, 2))

    # Векторизация для символов
    vectorized_x_train, vectorized_x_test = classify_with_char_vectorizer(x_train, x_test)

    # Шаг 4: Обучение модели логистической регрессии
    # clf = train_logistic_regression(vectorized_x_train, y_train)

    # Обучение через xgb_classifier
    # Преобразование меток в числовые значения
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    xgb_clf = train_xgb_classifier(vectorized_x_train, y_train)

    # Шаг 5: Оценка модели
    evaluate_model(xgb_clf, vectorized_x_test, y_test)
