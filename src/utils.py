import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

def load_data(df_train, df_test):
    print("Vectorizing...")
    vectorizer = CountVectorizer(ngram_range=(1, 3), binary=True)
    X_train = vectorizer.fit_transform(df_train.text)
    X_test = vectorizer.transform(df_test.text)
    print("Encoding Labels...")
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(df_train.label)
    y_test = le.transform(df_test.label)

    return X_train, X_test, y_train, y_test