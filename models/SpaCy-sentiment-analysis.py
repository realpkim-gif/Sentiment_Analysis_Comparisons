import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from preprocessing.clean import dataCleaning

# Loading dataset
data = pd.read_csv(r'dataFolder/train.csv', encoding="latin-1")
data = data[["text", "sentiment"]]
data = data.fillna("")

data["sentiment"] = data["sentiment"].map({"negative": 0, "positive": 1, "neutral": 2})

# Splitting the train and test data
X = data['text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creating the model and pipeline
tfidf = TfidfVectorizer(tokenizer=dataCleaning)
Lr = LogisticRegression()
steps = [('tfidf', tfidf), ('Lr', Lr)]
pipe = Pipeline(steps)

# Training the model
pipe.fit(X_train, y_train)

# Testing on the test dataset
y_pred = pipe.predict(X_test)

# Printing the classification report and the confusion matrix
print(classification_report(y_test, y_pred))
print("\n\n")
print(confusion_matrix(y_test, y_pred))
