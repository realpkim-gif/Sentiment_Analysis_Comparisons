import os

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def BertTopic(texts):
    # Collapse extra spaces/newlines so each review is one clean line of text
    docs = texts.str.replace(r"\s+", " ", regex=True).tolist()

    # Turn each review into a list of numbers (an "embedding") that captures its meaning
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=64)

    topic_model = BERTopic(
        embedding_model=embedder,
        vectorizer_model=CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2)),
        min_topic_size=50,
        nr_topics="auto",
        verbose=True,
    )
    pred, probs = topic_model.fit_transform(docs, embeddings)  # predicted topic for each review (-1 = outlier)

    topic_model.get_topic_info().to_csv("yelp_topics.csv", index=False)
    os.makedirs("plots", exist_ok=True)
    topic_model.visualize_barchart(top_n_topics=12).write_html("plots/topics_barchart.html")
    topic_model.visualize_topics().write_html("plots/topics_map.html")
    topic_model.visualize_hierarchy().write_html("plots/topics_hierarchy.html")

    # top 10 words for each topic (-1 is the outlier bucket, not a real topic)
    words = []
    for topic_id in topic_model.get_topics():
        if topic_id == -1:
            continue
        topic_words = topic_model.get_topic(topic_id)  # list of (word, score)
        words.append([word for word, score in topic_words])

    return {
        "pred": pred,
        "words": words,
        "analyzer": topic_model.vectorizer_model.build_analyzer(),
    }
