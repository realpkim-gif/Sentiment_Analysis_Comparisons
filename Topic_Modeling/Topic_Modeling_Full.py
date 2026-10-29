import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from Berttopic import BertTopic
from LDA import LDA_Model
from LSA import LSA_Model

# Run from the Topic_Modeling folder (it reads yelp_data/ and saves CSVs here)
N_REVIEWS = 30000
TOPIC_COUNTS = [5, 10, 15, 20, 30, 40, 50]


def load_reviews(n=N_REVIEWS):
    """Return the text of the first n restaurant reviews from the Yelp dataset."""
    # The Yelp data has two files: one listing businesses, one listing reviews.
    # Each review only has a business_id, so we use the business file to find which ones are restaurants.
    businesses = pd.read_json("yelp_data/yelp_academic_dataset_business.json", lines=True)
    is_restaurant = businesses["categories"].fillna("").str.contains("Restaurants")
    restaurant_ids = set(businesses.loc[is_restaurant, "business_id"])

    # The review file is huge (~5 GB), so read it in pieces and stop once we have enough restaurant reviews
    kept_pieces = []
    total_kept = 0
    for piece in pd.read_json("yelp_data/yelp_academic_dataset_review.json", lines=True, chunksize=100_000):
        piece = piece[piece["business_id"].isin(restaurant_ids)]  # keep only restaurant reviews
        kept_pieces.append(piece)
        total_kept += len(piece)
        if total_kept >= n:
            break

    reviews = pd.concat(kept_pieces).head(n)
    return reviews["text"].fillna("").reset_index(drop=True)


def coherence(words, texts, analyzer):
    """c_v coherence on 5000 sampled reviews, higher is better (do a topic's words show up together?)"""
    tokenized = []
    for review in texts.sample(min(5000, len(texts)), random_state=0):
        tokenized.append(analyzer(review))
    dictionary = Dictionary(tokenized)

    # gensim errors on words it has never seen, so drop those
    topics = []
    for topic in words:
        known = [word for word in topic if word in dictionary.token2id]
        if known:
            topics.append(known)

    cm = CoherenceModel(topics=topics, texts=tokenized, dictionary=dictionary, coherence="c_v", processes=1)
    return cm.get_coherence()


def diversity(words):
    """unique top words / total top words, 1.0 = no word repeats across topics, lower = more overlap"""
    all_words = []
    for topic in words:
        all_words.extend(topic)
    return len(set(all_words)) / len(all_words)


if __name__ == "__main__":
    texts = load_reviews()

    # Run every model. Each result is a (model name, output) pair.
    # LSA and LDA are tried with several topic counts; BERTopic chooses its own count.
    runs = []
    for k in TOPIC_COUNTS:
        runs.append(("LSA", LSA_Model(texts, k)))
        runs.append(("LDA", LDA_Model(texts, k)))
    runs.append(("BERTopic", BertTopic(texts)))

    # Build one table of scores (one row per run) and one table of each review's predicted topic
    rows = []
    predictions = pd.DataFrame({"text": texts})
    for name, run in runs:
        n_topics = len(run["words"])
        rows.append({
            "model": name,
            "n_topics": n_topics,
            "explained_variance": run.get("explained_variance"),  # LSA only, blank for others
            "perplexity": run.get("perplexity"),                  # LDA only, blank for others
            "coherence": coherence(run["words"], texts, run["analyzer"]),
            "diversity": diversity(run["words"]),
        })
        predictions[f"{name}_{n_topics}"] = run["pred"]

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv("model_metrics.csv", index=False) #key importance
    predictions.to_csv("topic_predictions.csv", index=False)
