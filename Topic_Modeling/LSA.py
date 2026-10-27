"""LSA topic modeling on Yelp restaurant reviews, sweeping n_components (number of topics).

Run:  python Topic_Modeling/LSA.py
"""
import os

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from LDA import load_reviews

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC_COUNTS = [5, 10, 15, 20, 30]


def LSA_Model():
    texts = load_reviews()

    # LSA works on TF-IDF weights (unlike LDA, which needs raw counts)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(texts)

    results = []
    for k in TOPIC_COUNTS:
        lsa = TruncatedSVD(n_components=k, random_state=0)
        lsa.fit(X)
        explained = lsa.explained_variance_ratio_.sum()  # higher is better, but always rises with k
        results.append({"n_components": k, "explained_variance": explained})
        print(f"\nn_components={k}  explained variance={explained:.3f}")

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(HERE, "lsa_explained_variance.csv"), index=False)
    return results


if __name__ == "__main__":
    print(LSA_Model())
