from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def train_logistic_regression(x_train, y_train):
    clf = LogisticRegression(random_state=24, max_iter=1000)
    clf.fit(x_train, y_train)
    return clf


def train_xgb_classifier(x_train, y_train, y_test):
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
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
    return xgb_clf, y_test


def train_random_forest(x_train, y_train):
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=24
    )
    rf_clf.fit(x_train, y_train)
    return rf_clf


def evaluate_model(clf, x_test, y_test):
    pred = clf.predict(x_test)
    print(classification_report(y_test, pred))
