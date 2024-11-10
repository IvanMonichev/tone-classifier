
from sklearn.model_selection import train_test_split

from data_loading import load_and_label_data
from models_traning import train_random_forest, evaluate_model
from vectorization import  vectorize_char

RANDOM_SEED = 42

if __name__ == '__main__':
    # Загрузка данных и форматирование их в датафрейм
    dataframe = load_and_label_data('./content/positive.csv', './content/negative.csv')

    # Разделение данных на тренировочные и тестовые выборки
    x_train, x_test, y_train, y_test = train_test_split(dataframe['text'], dataframe['label'], random_state=RANDOM_SEED)

    # === ВЕКТОРИЗАЦИЯ ===
    # Униграммы
    # vectorized_x_train, vectorized_x_test = vectorize_texts(x_train, x_test, (1, 1))
    # Триграммы
    # vectorized_x_train, vectorized_x_test = vectorize_texts(x_train, x_test, (3, 3))
    # # TF-IDF векторизация
    # vectorized_x_train, vectorized_x_test = t_fidf_vectorize_texts(x_train, x_test, (1, 1))
    # Умная векторизация с токенами
    # vectorized_x_train, vectorized_x_test = vectorize_texts_with_tokens_and_stopwords(x_train, x_test, (1, 2))
    # Векторизация без стоп-слов
    # vectorized_x_train, vectorized_x_test = vectorize_texts_without_stopwords(x_train, x_test, CountVectorizer, (1, 2))
    # Векторизация для символов
    vectorized_x_train, vectorized_x_test = vectorize_char(x_train, x_test, (1, 1))

    # === ОБУЧЕНИЕ ===
    # Обучение через логистическую регрессию
    # clf = train_logistic_regression(vectorized_x_train, y_train)
    # Обучение через XGBClassifier
    # clf, y_test = train_xgb_classifier(vectorized_x_train, y_train, y_test)
    # Обучение через случайный лес
    clf = train_random_forest(vectorized_x_train, y_train)

    evaluate_model(clf, vectorized_x_test, y_test)
