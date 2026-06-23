from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing.clean import dataCleaning

def get_accuracy(df):
    X = df['text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tfidf = TfidfVectorizer(tokenizer=dataCleaning)
    Lr = LogisticRegression()
    steps = [('tfidf', tfidf), ('Lr', Lr)]
    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    preds = list(y_pred)
    return ("SpaCy + LR", accuracy, preds)
