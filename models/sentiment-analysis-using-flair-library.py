import math
import re
import numpy as np
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence

filename = r"dataFolder/train.csv"
df = pd.read_csv(filename, encoding='latin-1')

oneFifth = math.ceil(len(df) * 0.05)
df = df["text"].sample(oneFifth)
df.reset_index(drop=True, inplace=True)
df = df.to_frame()

def remove_urls(x):
    cleaned_string = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', str(x), flags=re.MULTILINE)
    return cleaned_string

def deEmojify(x):
    regrex_pattern = re.compile(pattern="["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', x)

def unify_whitespaces(x):
    cleaned_string = re.sub(r'\s+', ' ', x)
    return cleaned_string

def remove_symbols(x):
    cleaned_string = re.sub(r"[^a-zA-Z0-9?!.,]+", ' ', x)
    return cleaned_string

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
