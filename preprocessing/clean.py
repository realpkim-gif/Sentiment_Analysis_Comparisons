import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd


# spacy setup
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = list(STOP_WORDS)
lemmatizer = WordNetLemmatizer()

punct = string.punctuation

abbreviations = pd.read_csv("dataFolder/abbrevations.csv")
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

# --- flair-style cleaning ---
def remove_urls(x):
    cleaned_string = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', str(x), flags=re.MULTILINE)
    return cleaned_string

def deEmojify(x):
    regrex_pattern = re.compile(pattern="["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', x)

def unify_whitespaces(x):
    cleaned_string = re.sub(r'\s+', ' ', x)
    return cleaned_string

def remove_symbols(x):
    cleaned_string = re.sub(r"[^a-zA-Z0-9?!.,]+", ' ', x)
    return cleaned_string


# --- vader/textblob-style cleaning ---

def strip_html(text):
    return re.sub(r"<[^>]+>", "", text)


# --- spacy-style cleaning (used as tokenizer for tfidf) ---

def dataCleaning(sentence):
    doc = nlp(sentence)

    clean_tokens = []
    for token in doc:
        temp = token.lemma_.lower().strip()
        if temp not in punct and temp not in spacy_stopwords:
            clean_tokens.append(temp)
        else:
            pass

    return clean_tokens


# --- sentiwordnet-style cleaning ---

def preprocess_data_general(data, name):
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

def lemmatization(data, name):
    def getting2(sen):
        if isinstance(sen, list):
            sen = ' '.join(sen)
        example = sen
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
