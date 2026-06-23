import math
import numpy as np
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from preprocessing.clean import remove_urls, deEmojify, remove_symbols, unify_whitespaces

filename = r"dataFolder/train.csv"
df = pd.read_csv(filename, encoding='latin-1')

oneFifth = math.ceil(len(df) * 0.05)
df = df["text"].sample(oneFifth)
df.reset_index(drop=True, inplace=True)
df = df.to_frame()

df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(remove_urls)
df['text'] = df['text'].apply(deEmojify)
df['text'] = df['text'].apply(remove_symbols)
df['text'] = df['text'].apply(unify_whitespaces)

sia = TextClassifier.load('en-sentiment')

def flair_prediction(text):
    sentence = Sentence(text)
    sia.predict(sentence)
    label = sentence.labels[0].value
    score = sentence.labels[0].score
    return label, score

df['Sentiment'] = ""
df['Sentiment_Score'] = np.nan

df['Sentiment'], df['Sentiment_Score'] = zip(*df['text'].apply(flair_prediction))

print(df.head())
print(df.groupby(["Sentiment"]).agg({'Sentiment_Score': ['count', 'mean']}))
