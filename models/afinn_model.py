from afinn import Afinn

afinn = Afinn(language='en')

def afinn_to_sentiment(score):
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

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
    return ("AFINN", f"{accuracy:.2%}")
