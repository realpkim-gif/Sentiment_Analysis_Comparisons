from nltk.sentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()


def vader_sentiment(text):
    score = vader.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def predict(df):
    return df["clean_text"].apply(vader_sentiment)
