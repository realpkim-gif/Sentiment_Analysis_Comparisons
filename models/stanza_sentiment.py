import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #just to fix glitch

import stanza
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,sentiment', use_gpu=True)

data = pd.read_csv("dataFolder/train.csv", encoding="latin-1")

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


def scoring(inference, label):
    accuracy = accuracy_score(label, inference)
    f1 = f1_score(label, inference, average="weighted")
    precision = precision_score(label, inference, average="weighted", zero_division=0)
    recall = recall_score(label, inference, average="weighted")
    return accuracy, f1, precision, recall

true_to_num()

print(data.shape)
print(len(output))
print(scoring(output,num_labels))
print("Ultimately, still not very accurate but should be more accurate than Vader as it is Neural network based. Because it is CNN based, it can understand more complex wording like (not bad) being positive. While Vader only looks at simple rules and looks up words in its lexicon, Stanza has a pretrained CNN that can understand complex wording and understand how words affect other words in a given context."
      "However because Stanza was trained on formal text (news articles, books) not Twitter/social media. "
      "Slang, abbreviations, emojis, and sarcasm all mix it up. This data has a lot of that")
