import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #just to fix glitch

import re
import stanza
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing.clean import deEmojify, remove_symbols, unify_whitespaces

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,sentiment', use_gpu=True)

data = pd.read_csv("dataFolder/train.csv", encoding="latin-1")
data = data[data["text"].str.strip() != ""]

data = data.dropna(subset=["text"])
data = data[data["text"].str.strip() != ""]
data["text"] = data["text"].fillna("").astype(str)

#extra preprocessing
data["text"] = data["text"].apply(deEmojify)
data["text"] = data["text"].apply(remove_symbols)
data["text"] = data["text"].apply(unify_whitespaces)
data["text"] = data["text"].apply(lambda x: re.sub(r'(.)\1{2,}', r'\1\1', x))

data = data.dropna(subset=["text"])
data = data[data["text"].str.strip() != ""]
data["text"] = data["text"].fillna("").astype(str)


texts = data['text']
texts=texts.to_list()

labels = data['sentiment']

output=[]
num_labels=[]
docs = nlp.bulk_process([str(i) for i in texts])

for doc in docs:
    sentiments = [s.sentiment for s in doc.sentences]
    output.append(round(sum(sentiments) / len(sentiments)))

def true_to_num():
    for i in labels:
        if i == "positive":
            num_labels.append(2)
        elif i == "negative":
            num_labels.append(0)
        elif i == "neutral":
            num_labels.append(1)
        else:
            num_labels.append(1)


true_to_num()


def get_accuracy():
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    preds = [label_map[p] for p in output]
    acc = accuracy_score(num_labels, output)
    return ("Stanza", acc, preds)

print("Stanza:")
print(get_accuracy())
print("Stanza lib: Ultimately, still not as accurate as expected but should be more accurate than Vader as it is Neural network based. Because it is CNN based, it can understand more complex wording like (not bad) being positive. While Vader only looks at simple rules and looks up words in its lexicon, Stanza has a pretrained CNN that can understand complex wording and understand how words affect other words in a given context."
      "However because Stanza was trained on formal text (news articles, books) not Twitter/social media. "
      "Slang, abbreviations, emojis, and sarcasm all mix it up. This data has a lot of that")
