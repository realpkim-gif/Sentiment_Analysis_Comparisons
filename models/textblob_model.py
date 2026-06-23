from textblob import TextBlob
from preprocessing.clean import strip_html

def textblob_sentiment(text):
    text = strip_html(text)
    score = TextBlob(text).sentiment.polarity

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def get_accuracy(df):
    df["textblob_pred"] = df["text"].apply(textblob_sentiment)

    correct = 0
    total = 0
    for i in range(len(df)):
        total += 1
        if df["sentiment"].iloc[i] == df["textblob_pred"].iloc[i]:
            correct += 1

    accuracy = correct / total
    preds = list(df["textblob_pred"])
    return ("TextBlob", accuracy, preds)
