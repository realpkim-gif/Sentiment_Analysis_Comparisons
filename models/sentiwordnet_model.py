import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from preprocessing.clean import preprocess_data_general, rem_stopwords_tokenize, lemmatization, make_sentences

lemmatizer = WordNetLemmatizer()

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_sentiment(word, tag):
    wn_tag = penn_to_wn(tag)

    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [synset.name(), swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()]

def get_accuracy(df):
    data = df.copy()
    data["text_without_stopwords"] = data["text"].copy()
    data["text_without_stopwords"] = data["text_without_stopwords"].fillna('').astype(str)

    preprocess_data_general(data, 'text_without_stopwords')
    rem_stopwords_tokenize(data, 'text_without_stopwords')
    make_sentences(data, 'text_without_stopwords')

    data["After_lemmatization"] = data['text_without_stopwords'].copy()
    lemmatization(data, 'After_lemmatization')
    make_sentences(data, 'After_lemmatization')

    postagging = []
    for review in data['After_lemmatization']:
        list = word_tokenize(review)
        postagging.append(nltk.pos_tag(list))
    data['pos_tags'] = postagging

    pos = neg = 0
    senti_score = []
    for pos_val in data['pos_tags']:
        senti_val = [get_sentiment(x, y) for (x, y) in pos_val]
        for score in senti_val:
            try:
                pos = pos + score[1]
                neg = neg + score[2]
            except:
                continue
        senti_score.append(pos - neg)
        pos = neg = 0
    data['senti_score'] = senti_score

    overall = []
    for i in range(len(data)):
        if data['senti_score'][i] >= 0.05:
            overall.append('positive')
        elif data['senti_score'][i] <= -0.05:
            overall.append('negative')
        else:
            overall.append('neutral')
    data['Overall Sentiment'] = overall

    accuracy = accuracy_score(data["sentiment"], data["Overall Sentiment"])
    preds = list(data["Overall Sentiment"])
    return ("SentiWordNet", accuracy, preds)
