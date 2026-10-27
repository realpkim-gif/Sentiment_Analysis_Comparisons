"""LDA topic modeling on Yelp restaurant reviews, sweeping n_components (number of topics).

Run:  python Topic_Modeling/LDA.py
"""
import os

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "yelp_data")
N_REVIEWS = 30_000
TOPIC_COUNTS = [5, 10, 15, 20, 30]


def load_reviews(n=N_REVIEWS):
    biz = pd.read_json(os.path.join(DATA, "yelp_academic_dataset_business.json"), lines=True)
    rest_ids = set(biz.loc[biz["categories"].fillna("").str.contains("Restaurants"), "business_id"])

    parts, total = [], 0
    for chunk in pd.read_json(os.path.join(DATA, "yelp_academic_dataset_review.json"), lines=True, chunksize=100_000):
        chunk = chunk[chunk["business_id"].isin(rest_ids)]
        parts.append(chunk)
        total += len(chunk)
        if total >= n:
            break
    return pd.concat(parts).head(n)["text"].fillna("")


def LDA_Model():
    texts = load_reviews()

    # LDA works on raw word counts (not TF-IDF); drop words that appear in nearly every review
    vectorizer = CountVectorizer(stop_words="english", min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(texts)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
    feature_names = vectorizer.get_feature_names_out()

    results = []
    for k in TOPIC_COUNTS:
        lda = LatentDirichletAllocation(n_components=k, learning_method="online", random_state=0, n_jobs=-1)
        lda.fit(X_train)
        perplexity = lda.perplexity(X_test)  # held-out; lower is better
        results.append({"n_components": k, "perplexity": perplexity})
        print(f"\nn_components={k}  held-out perplexity={perplexity:.1f}")

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(HERE, "lda_perplexity.csv"), index=False)
    return results


if __name__ == "__main__":
    print(LDA_Model())
