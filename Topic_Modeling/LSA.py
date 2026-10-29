from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def LSA_Model(texts, k):
    # LSA works on TF-IDF weights (unlike LDA, which needs raw counts)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(texts)

    lsa = TruncatedSVD(n_components=k, random_state=0)
    lsa.fit(X)

    # top 10 words for each topic
    vocab = vectorizer.get_feature_names_out()
    words = []
    for topic_weights in lsa.components_:
        top_ids = topic_weights.argsort()[::-1][:10]  # word positions, highest weight first
        words.append([vocab[i] for i in top_ids])

    return {
        "pred": lsa.transform(X).argmax(axis=1),  # predicted topic for each review
        "words": words,
        "analyzer": vectorizer.build_analyzer(),
        "explained_variance": lsa.explained_variance_ratio_.sum(),  # higher is better, but always rises with k
    }
