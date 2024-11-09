import pandas
from sklearn.model_selection import train_test_split


def create_tweet_dataframe():
    positive = pandas.read_csv('./content/positive.csv', sep=';', usecols=[3], names=['text'])
    negative = pandas.read_csv('./content/negative.csv', sep=';', usecols=[3], names=['text'])

    positive['label'] = ['positive'] * len(positive)
    negative['label'] = ['negative'] * len(negative)

    return pandas.concat([positive, negative])


def split_data(texts, labels):
    x_train, x_test, y_train, y_test = train_test_split(texts, labels)

if __name__ == '__main__':
    dataframe = create_tweet_dataframe()
    split_data(dataframe['text'], dataframe['label'])
