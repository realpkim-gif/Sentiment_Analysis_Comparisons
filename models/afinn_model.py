from afinn import Afinn
import pandas as pd

afinn = Afinn(language='en')

data = pd.read_csv("dataFolder/train.csv", encoding="latin-1")
data = data.dropna(subset=["text"])

scores = []
for i in data["text"]:
    scores.append(afinn.score(i))

data["affin score"] = scores

def afinn_to_sentiment(score):
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

data["afinn_pred"] = data["affin score"].apply(afinn_to_sentiment)

def get_accuracy(df):
    scores = []
    for i in df["text"]:
        scores.append(afinn.score(i))

    df["affin score"] = scores
    df["afinn_pred"] = df["affin score"].apply(afinn_to_sentiment)

    correct = 0
    total = 0
    for i in range(len(df)):
        total += 1
        if df["sentiment"].iloc[i] == df["afinn_pred"].iloc[i]:
            correct += 1

    accuracy = correct / total
    preds = list(df["afinn_pred"])
    return ("AFINN", accuracy, preds)
