from afinn import Afinn

afinn = Afinn(language='en')

def afinn_to_sentiment(score):
    if score > 1:
        return "positive"
    elif score < -1:
        return "negative"
    else:
        return "neutral"

def predict(df):
    return df["clean_text"].apply(lambda x: afinn_to_sentiment(afinn.score(x)))
