import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flair.models import TextClassifier
from flair.data import Sentence
from preprocessing.clean import remove_urls, deEmojify, remove_symbols, unify_whitespaces

sia = TextClassifier.load('en-sentiment')

def flair_prediction(text):
    try:
        sentence = Sentence(str(text))
        sia.predict(sentence)
        return sentence.labels[0].value.lower(), sentence.labels[0].score
    except:
        return None, None

def get_accuracy(df):
    # flair is slow so use 5% sample
    oneFifth = math.ceil(len(df) * 0.05)
    df_sample = df[df["sentiment"] != "neutral"].sample(oneFifth).copy()

    df_sample['text'] = df_sample['text'].str.lower()
    df_sample['text'] = df_sample['text'].apply(remove_urls)
    df_sample['text'] = df_sample['text'].apply(deEmojify)
    df_sample['text'] = df_sample['text'].apply(remove_symbols)
    df_sample['text'] = df_sample['text'].apply(unify_whitespaces)

    df_sample["flair_pred"], df_sample["confidence"] = zip(*df_sample["text"].apply(flair_prediction))

    correct = 0
    total = 0
    for i in range(len(df_sample)):
        total += 1
        if df_sample["sentiment"].iloc[i] == df_sample["flair_pred"].iloc[i]:
            correct += 1

    accuracy = correct / total
    return ("Flair (5% pos/neg)", f"{accuracy:.2%}")
