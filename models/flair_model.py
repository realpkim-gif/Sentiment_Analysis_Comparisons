import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flair.models import TextClassifier
from flair.data import Sentence
from preprocessing.clean import remove_urls, deEmojify, remove_symbols, unify_whitespaces

sia = TextClassifier.load('en-sentiment')

def flair_prediction(text):
    try:
        sentence = Sentence(str(text))
        sia.predict(sentence)
        return sentence.labels[0].value.lower(), sentence.labels[0].score
    except:
        return None, None

def get_accuracy(df):
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(remove_urls)
    df['text'] = df['text'].apply(deEmojify)
    df['text'] = df['text'].apply(remove_symbols)
    df['text'] = df['text'].apply(unify_whitespaces)

    df["flair_pred"], df["confidence"] = zip(*df["text"].apply(flair_prediction))

    correct = 0
    total = 0
    for i in range(len(df)):
        total += 1
        if df["sentiment"].iloc[i] == df["flair_pred"].iloc[i]:
            correct += 1

    accuracy = correct / total
    preds = list(df["flair_pred"])
    return ("Flair", accuracy, preds)
