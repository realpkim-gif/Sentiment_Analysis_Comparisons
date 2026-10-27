import os
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd


# spacy setup
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = set(STOP_WORDS)

punct = string.punctuation

abbreviations = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataFolder", "abbrevations.csv"))
abbrev_dict = dict(zip(abbreviations["slang"].str.lower(), abbreviations["meaning"].str.lower()))

def apply_abbreviations(text):
    words = text.lower().split()
    result = []
    for word in words:
        if word in abbrev_dict:
            result.append(abbrev_dict[word])
        else:
            result.append(word)
    return " ".join(result)


# --- the one function every model uses ---
# clean_text -> VADER, TextBlob, AFINN, Stanza, Flair
# lemma_text -> NRC, SentiWordNet, SpaCy + LR

def preprocess(df):
    text = df["text"].astype(str)
    text = text.str.replace(r"<[^>]+>", "", regex=True)                                        # html tags
    text = text.str.replace(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', regex=True)    # urls
    text = text.str.replace("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", '', regex=True)  # emojis
    text = text.str.replace(r"[^a-zA-Z0-9?!.,]+", ' ', regex=True)                             # symbols
    text = text.str.replace(r'\s+', ' ', regex=True)                                           # whitespace
    text = text.str.replace(r'(.)\1{2,}', r'\1\1', regex=True)                                 # sooooo -> soo
    df["clean_text"] = text.str.strip()

    lemmas = []
    for doc in nlp.pipe(df["clean_text"], batch_size=256):
        tokens = [token.lemma_.lower().strip() for token in doc]
        lemmas.append(" ".join(t for t in tokens if t not in punct and t not in spacy_stopwords))
    df["lemma_text"] = lemmas
    return df
