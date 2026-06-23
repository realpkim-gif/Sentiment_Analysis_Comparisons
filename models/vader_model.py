from nltk.sentiment import SentimentIntensityAnalyzer
from preprocessing.clean import strip_html

vader = SentimentIntensityAnalyzer()


def vader_sentiment(text):
    text=strip_html(text)
    score = vader.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def get_accuracy(df):
    df["vader"] = df["text"].apply(lambda x: vader_sentiment(x))

    correct = 0
    total = 0
    for i in range(len(df)):
        total += 1
        if df["sentiment"].iloc[i] == df["vader"].iloc[i]:
            correct += 1

    accuracy = correct / total
    return ("VADER", f"{accuracy:.2%}")
