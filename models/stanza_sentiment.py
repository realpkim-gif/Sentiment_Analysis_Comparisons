import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #just to fix glitch

import stanza
import pandas as pd

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,sentiment', use_gpu=True)


def predict(df):
    docs = nlp.bulk_process([str(i) for i in df["clean_text"]])

    output = []
    for doc in docs:
        sentiments = [s.sentiment for s in doc.sentences]
        output.append(round(sum(sentiments) / len(sentiments)) if sentiments else 1)

    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return pd.Series([label_map[p] for p in output], index=df.index)

"""
Stanza lib: Ultimately, still not as accurate as expected but should be more accurate than Vader as it is Neural network based. Because it is CNN based, it can understand more complex wording like (not bad) being positive. While Vader only looks at simple rules and looks up words in its lexicon, Stanza has a pretrained CNN that can understand complex wording and understand how words affect other words in a given context.
However because Stanza was trained on formal text (news articles, books) not Twitter/social media. Slang, abbreviations, emojis, and sarcasm all mix it up. This data has a lot of that
"""
