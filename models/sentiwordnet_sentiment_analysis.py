import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

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

# Using LogReg with TF-IDF is significantly better than SentiWordNet alone (51% -> ~79%)
def predict(df):
    data2 = df.copy()

    postagging = []
    for review in data2['lemma_text']:
        tokens = word_tokenize(review)
        postagging.append(nltk.pos_tag(tokens))
    data2['pos_tags'] = postagging

    pos = neg = 0
    senti_score = []
    for pos_val in data2['pos_tags']:
        senti_val = [get_sentiment(x, y) for (x, y) in pos_val]
        for score in senti_val:
            try:
                pos = pos + score[1]
                neg = neg + score[2]
            except:
                continue
        senti_score.append(pos - neg)
        pos = neg = 0
    data2['senti_score'] = senti_score

    overall = []
    for i in range(len(data2)):
        if data2['senti_score'][i] >= 0.05:
            overall.append('positive')
        elif data2['senti_score'][i] <= -0.05:
            overall.append('negative')
        else:
            overall.append('neutral')
    return pd.Series(overall, index=data2.index)
