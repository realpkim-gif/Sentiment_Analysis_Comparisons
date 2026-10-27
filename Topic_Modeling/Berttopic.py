import os

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "yelp_data")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

N_REVIEWS = 30_000  # keep small for a first run; raise once it works

biz = pd.read_json(os.path.join(DATA, "yelp_academic_dataset_business.json"), lines=True)
rest_ids = set(biz.loc[biz["categories"].fillna("").str.contains("Restaurants"), "business_id"])

# take the first reviews from the file that belong to restaurants (avoids reading all 5 GB)
parts, total = [], 0
for chunk in pd.read_json(os.path.join(DATA, "yelp_academic_dataset_review.json"), lines=True, chunksize=100_000):
    chunk = chunk[chunk["business_id"].isin(rest_ids)]
    parts.append(chunk)
    total += len(chunk)
    if total >= N_REVIEWS:
        break
reviews = pd.concat(parts).head(N_REVIEWS)
docs = reviews["text"].str.replace(r"\s+", " ", regex=True).tolist()
print(f"{len(docs):,} restaurant reviews")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=64)

topic_model = BERTopic(
    embedding_model=embedder,
    vectorizer_model=CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2)),
    min_topic_size=50,
    nr_topics="auto",
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings)

info = topic_model.get_topic_info()
print(info.head(20))
info.to_csv(os.path.join(HERE, "yelp_topics.csv"), index=False)

topic_model.visualize_barchart(top_n_topics=12).write_html(os.path.join(OUT, "topics_barchart.html"))
topic_model.visualize_topics().write_html(os.path.join(OUT, "topics_map.html"))
topic_model.visualize_hierarchy().write_html(os.path.join(OUT, "topics_hierarchy.html"))
