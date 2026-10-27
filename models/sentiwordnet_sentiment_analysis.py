import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocessing.clean import preprocess_data_general, rem_stopwords_tokenize, lemmatization, make_sentences

lemmatizer = WordNetLemmatizer()

data = pd.read_csv('dataFolder/train.csv', encoding="latin-1")
data = data.drop(columns=["textID", "selected_text", "Time of Tweet", "Age of User", "Country", "Population -2020", "Land Area (Km²)", "Density (P/Km²)"])
data["text"] = data["text"].fillna('')

# Replacing Positive -> 1 and Negative -> 0
data.replace({"positive": 1, "negative": 0}, inplace=True)

#Edits After Removing Stopwords
Edited_Review = data['text'].copy()
data['text_without_stopwords'] = Edited_Review

data["text_without_stopwords"] = data["text_without_stopwords"].fillna('').astype(str)

preprocess_data_general(data, 'text_without_stopwords')
rem_stopwords_tokenize(data, 'text_without_stopwords')
make_sentences(data, 'text_without_stopwords')

final_Edit = data['text_without_stopwords'].copy()
data["After_lemmatization"] = final_Edit

lemmatization(data, 'After_lemmatization')
make_sentences(data, 'After_lemmatization')

pos = neg = obj = count = 0

postagging = []
for review in data['After_lemmatization']:
    tokens = word_tokenize(review)
    postagging.append(nltk.pos_tag(tokens))

data['pos_tags'] = postagging

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
        overall.append('Positive')
    elif data['senti_score'][i] <= -0.05:
        overall.append('Negative')
    else:
        overall.append('Neutral')
data['Overall Sentiment'] = overall
data['Overall Sentiment'] = data['Overall Sentiment'].str.lower()

# Replacing back 1 -> positive and 0 -> negative
data.replace({1: "positive", 0: "negative"}, inplace=True)

data['reviews_text_new'] = data['After_lemmatization'].copy()

# Bag-of-words
bow_counts = CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))
bow_data = bow_counts.fit_transform(data['reviews_text_new'])

X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_data,
                                                                    data['Overall Sentiment'],
                                                                    test_size=0.2,
                                                                    random_state=0)

lr_model_all = LogisticRegression(max_iter=1000)
lr_model_all.fit(X_train_bow, y_train_bow)
test_pred_lr_all = lr_model_all.predict(X_test_bow)
print(classification_report(y_test_bow, test_pred_lr_all))

# TF-IDF model
noise_words = []
tfidf_counts = TfidfVectorizer(tokenizer=word_tokenize, stop_words=noise_words, ngram_range=(1, 1))
tfidf_data = tfidf_counts.fit_transform(data['reviews_text_new'])

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_data,
                                                                            data['Overall Sentiment'],
                                                                            test_size=0.2,
                                                                            random_state=0)

lr_model_tf_idf = LogisticRegression()
lr_model_tf_idf.fit(X_train_tfidf, y_train_tfidf)
test_pred_lr_all = lr_model_tf_idf.predict(X_test_tfidf)
print(classification_report(y_test_tfidf, test_pred_lr_all))

# Using LogReg with TF-IDF is significantly better than SentiWordNet alone (51% -> ~79%)
def get_accuracy(df):
    from sklearn.metrics import accuracy_score
    data2 = df.copy()
    data2["text_without_stopwords"] = data2["text"].copy()
    data2["text_without_stopwords"] = data2["text_without_stopwords"].fillna('').astype(str)

    preprocess_data_general(data2, 'text_without_stopwords')
    rem_stopwords_tokenize(data2, 'text_without_stopwords')
    make_sentences(data2, 'text_without_stopwords')

    data2["After_lemmatization"] = data2['text_without_stopwords'].copy()
    lemmatization(data2, 'After_lemmatization')
    make_sentences(data2, 'After_lemmatization')

    postagging = []
    for review in data2['After_lemmatization']:
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
    data2['Overall Sentiment'] = overall

    accuracy = accuracy_score(data2["sentiment"], data2["Overall Sentiment"])
    preds = list(data2["Overall Sentiment"])
    return ("SentiWordNet", accuracy, preds)
