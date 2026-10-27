from afinn import Afinn
import pandas as pd
from preprocessing.clean import remove_urls, deEmojify, remove_symbols, unify_whitespaces

afinn = Afinn(language='en')

data = pd.read_csv("dataFolder/train.csv", encoding="latin-1")
data = data.dropna(subset=["text"])

data["text"] = data["text"].apply(remove_urls)
data["text"] = data["text"].apply(deEmojify)
data["text"] = data["text"].apply(remove_symbols)
data["text"] = data["text"].apply(unify_whitespaces)

def get_accuracy(df):

    def afinn_to_sentiment(score):
        if score > 1:
            return "positive"
        elif score < -1:
            return "negative"
        else:
            return "neutral"

    scores = []
    for i in df["text"]:
        scores.append(afinn.score(i))
    df["afinn score"] = scores
    df["afinn_pred"] = df["afinn score"].apply(afinn_to_sentiment)

    correct = 0
    total = 0
    for i in range(len(df)):
        total += 1
        if df["sentiment"].iloc[i] == df["afinn_pred"].iloc[i]:
            correct += 1

    accuracy = correct / total
    preds = list(df["afinn_pred"])
    return ("AFINN", accuracy, preds)

