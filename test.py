import textblob
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv(r'/dataFolder\train.csv', encoding="latin-1")

text = df["text"]
sentiment = df["sentiment"]

# Check for nulls first
print(df["text"].isna().sum())

df.loc[df["text"].isna()]

print(df.shape)

# Fix by dropping or filling
df["text"] = df["text"].fillna("")

print(df.shape)
print(df["text"].isna().sum())

stop_words = set(stopwords.words('english'))

def remove_punct(text):
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove @mentions
    text = re.sub(r'(.)\1{2,}', r'\1', text)     # sooo -> so, lolll -> lol

    # expand contractions before stripping punctuation
    text = re.sub(r"n[`']t", " not", text)        # couldn't/couldn`t -> could not
    text = re.sub(r"[`']re", " are", text)
    text = re.sub(r"[`']ve", " have", text)
    text = re.sub(r"[`']ll", " will", text)
    text = re.sub(r"[`']d",  " would", text)
    text = re.sub(r"[`']s",  "", text)

    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    text_nopunct = text_nopunct.lower()

    stop_words = set(stopwords.words('english'))
    tokens = re.split('\W+', text_nopunct)
    filtered = [w for w in tokens if w not in stop_words]

    return filtered

df['body_text_clean'] = df['text'].apply(lambda x: remove_punct(x))

def lemmetize(text):
    wn = nltk.WordNetLemmatizer()
    lemmatized = []
    for i in text:
        lemm = wn.lemmatize(i)
        lemmatized.append(lemm)
    return lemmatized

df['body_text_full_clean'] = df['body_text_clean'].apply(lambda x: lemmetize(x))

def strip_html(text):
    return re.sub(r"<[^>]+>", "", text)

df["no_html"] = df["text"].apply(strip_html)

vader = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = vader.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["vader"] = df["text"].apply(lambda x: vader_sentiment(x))

def textblob_sentiment(text):
    score = TextBlob(text).sentiment.polarity

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["textblob_pred"] = df["text"].apply(textblob_sentiment)

def evaluate_model(true_labels, predicted_labels, model_name="Model"):
    accuracy  = accuracy_score(true_labels, predicted_labels)
    f1        = f1_score(true_labels, predicted_labels, average="weighted")
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall    = recall_score(true_labels, predicted_labels, average="weighted")

    print(f"--- {model_name} ---")
    print(f"Accuracy  : {accuracy:.2%}")
    print(f"F1 Score  : {f1:.2%}")
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print()

evaluate_model(df["sentiment"], df["textblob_pred"], "TextBlob")
evaluate_model(df["sentiment"], df["vader"],    "VADER")

def count(model):
    for label in df["sentiment"].unique():
        correct = sum((df["sentiment"] == label) & (df[model] == label))
        total = sum(df["sentiment"] == label)

        print(f"{label} — {correct}/{total}")

print("Vader:")
count("vader")

print("TextBlob:")
count("textblob_pred")

df["compound"] = df["text"].apply(lambda x: vader.polarity_scores(x)["compound"])
print(df["compound"].value_counts().head(10))

print(df[df["compound"] == 0.0]["text"].head(10))
