from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def LDA_Model(texts, k):
    # LDA works on raw word counts (not TF-IDF); drop words that appear in nearly every review
    vectorizer = CountVectorizer(stop_words="english", min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(texts)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

    lda = LatentDirichletAllocation(n_components=k, learning_method="online", random_state=0, n_jobs=-1)
    lda.fit(X_train)

    # top 10 words for each topic
    vocab = vectorizer.get_feature_names_out()
    words = []
    for topic_weights in lda.components_: #model topics
        top_ids = topic_weights.argsort()[::-1][:10]  # word positions, highest weight first
        words.append([vocab[i] for i in top_ids])

    return {
        "pred": lda.transform(X).argmax(axis=1),  # predicted topic for each review
        "words": words,
        "analyzer": vectorizer.build_analyzer(),
        "perplexity": lda.perplexity(X_test),  # held-out; lower is better
    }
