import math
import numpy as np
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from preprocessing.clean import remove_urls, deEmojify, remove_symbols, unify_whitespaces

df = pd.read_csv(r"dataFolder/train.csv", encoding='latin-1')
df = df.dropna(subset=['text'])
df = df[df["sentiment"].isin(["positive", "negative"])] #only predict pos and neg since it only can predict pos and neg, Scoring on neutral would be unfair

df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(remove_urls)
df['text'] = df['text'].apply(deEmojify)
df['text'] = df['text'].apply(remove_symbols)
df['text'] = df['text'].apply(unify_whitespaces)

df = df.dropna(subset=['text'])
df = df[df["text"].str.strip() != ""]
df["text"] = df["text"].fillna("").astype(str)


sia = TextClassifier.load('en-sentiment')

def flair_prediction(text):
    try:
        sentence = Sentence(text)
        sia.predict(sentence)
        label = sentence.labels[0].value
        score = sentence.labels[0].score

        """
        label — a single string, either "POSITIVE" or "NEGATIVE" (Flair outputs uppercase)
        score — a single float between 0.0 and 1.0 (confidence score, e.g. 0.9832)
        flair_prediction output — a tuple: ("POSITIVE", 0.9832)
        """
    except:
        label = "Neutral"
        score = 0
    return label, score

df['Sentiment'] = ""
df['Sentiment_Score'] = np.nan

df['Sentiment'], df['Sentiment_Score'] = zip(*df['text'].apply(flair_prediction))

print(df.head())
print(df.groupby(["Sentiment"]).agg({'Sentiment_Score': ['count', 'mean']}))

def get_accuracy(df):
    df = df.copy()
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(remove_urls)
    df['text'] = df['text'].apply(deEmojify)
    df['text'] = df['text'].apply(remove_symbols)
    df['text'] = df['text'].apply(unify_whitespaces)

    df['Sentiment'], df['Sentiment_Score'] = zip(*df['text'].apply(flair_prediction))
    df['Sentiment'] = df['Sentiment'].str.lower()

    correct = 0
    total = 0
    for i in range(len(df)):
        total += 1
        if df["sentiment"].iloc[i] == df["Sentiment"].iloc[i]:
            correct += 1

    accuracy = correct / total
    preds = list(df["Sentiment"])
    return ("Flair", accuracy, preds)

