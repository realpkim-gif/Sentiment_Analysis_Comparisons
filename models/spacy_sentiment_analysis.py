import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def predict(df):
    X = df['lemma_text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tfidf = TfidfVectorizer()
    Lr = LogisticRegression(max_iter=1000)
    steps = [('tfidf', tfidf), ('Lr', Lr)]
    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)
    return pd.Series(pipe.predict(X_test), index=X_test.index)
