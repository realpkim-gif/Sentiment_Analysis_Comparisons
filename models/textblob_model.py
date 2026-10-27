from textblob import TextBlob

def textblob_sentiment(text):
    score = TextBlob(text).sentiment.polarity

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def predict(df):
    return df["clean_text"].apply(textblob_sentiment)
