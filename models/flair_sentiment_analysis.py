import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence

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

def predict(df):
    labels, scores = zip(*df['clean_text'].apply(flair_prediction))
    return pd.Series(labels, index=df.index).str.lower()
