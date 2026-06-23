import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import spacy
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('dataFolder/sentiment_analysis.csv', encoding="latin-1")
data = data.drop(columns=["textID", "selected_text", "Time of Tweet", "Age of User", "Country", "Population -2020", "Land Area (Km²)", "Density (P/Km²)"])
data["text"] = data["text"].fillna('')

# Replacing Positive -> 1 and Negative -> 0
data.replace({"positive": 1, "negative": 0}, inplace=True)

#Edits After Removing Stopwords
Edited_Review = data['text'].copy()
data['text_without_stopwords'] = Edited_Review

def preprocess_Reviews_data(data, name):
    data[name] = data[name].str.lower()
    data[name] = data[name].apply(lambda x: re.sub(r'\B#\S+', '', x))
    data[name] = data[name].apply(lambda x: re.sub(r"http\S+", "", x))
    data[name] = data[name].apply(lambda x: ' '.join(re.findall(r"[\w']+", x)))
    data[name] = data[name].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))
    data[name] = data[name].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    data[name] = data[name].apply(lambda x: re.sub('@[^\s]+', ' ', x))

def rem_stopwords_tokenize(data, name):
    def getting(sen):
        example_sent = sen
        filtered_sentence = []
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(example_sent)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return filtered_sentence
    x = []
    for i in data[name].values:
        x.append(getting(i))
    data[name] = x

lemmatizer = WordNetLemmatizer()
def Lemmatization(data, name):
    def getting2(sen):
        if isinstance(sen, list):
            sen = ' '.join(sen)
        example = sen
        output_sentence = []
        word_tokens2 = word_tokenize(example)
        lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens2]
        without_single_chr = [word for word in lemmatized_output if len(word) > 2]
        cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
        return cleaned_data_title
    x = []
    for i in data[name].values:
        x.append(getting2(i))
    data[name] = x

def make_sentences(data, name):
    data[name] = data[name].apply(lambda x: ' '.join([i for i in x]))
    data[name] = data[name].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))

data["text_without_stopwords"] = data["text_without_stopwords"].fillna('').astype(str)

preprocess_Reviews_data(data, 'text_without_stopwords')
rem_stopwords_tokenize(data, 'text_without_stopwords')
make_sentences(data, 'text_without_stopwords')

final_Edit = data['text_without_stopwords'].copy()
data["After_lemmatization"] = final_Edit

Lemmatization(data, 'After_lemmatization')
make_sentences(data, 'After_lemmatization')

pos = neg = obj = count = 0

postagging = []
for review in data['After_lemmatization']:
    list = word_tokenize(review)
    postagging.append(nltk.pos_tag(list))

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

print(classification_report(data["Overall Sentiment"], data["sentiment"]))

data['reviews_text_new'] = data['After_lemmatization'].copy()

# Bag-of-words
bow_counts = CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))
bow_data = bow_counts.fit_transform(data['reviews_text_new'])

X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_data,
                                                                    data['Overall Sentiment'],
                                                                    test_size=0.2,
                                                                    random_state=0)

lr_model_all = LogisticRegression()
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
