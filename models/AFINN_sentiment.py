from afinn import Afinn
import pandas as pd
import numpy as np

afinn = Afinn(language='en')

data = pd.read_csv("dataFolder/train.csv", encoding="latin-1")
data = data.dropna(subset=["text"])

data[["text", "sentiment"]].head()

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

def evaluate(true, pred):
    correct = 0
    wrong = 0
    total = 0

    for i in range(len(true)):
        total += 1
        if true[i] == pred[i]:
            correct += 1
        else:
            wrong += 1

    accuracy = correct / total
    print(f"Correct: {correct}/{total}")
    print(f"Wrong  : {wrong}/{total}")
    print(f"Accuracy: {accuracy:.2%}")

evaluate(list(data["sentiment"]), list(data["afinn_pred"]))
