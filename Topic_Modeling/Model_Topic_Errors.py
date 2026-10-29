import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Run from the Topic_Modeling folder. Needs model_errors.csv from GetErrorsAndCorrection.py
errors = pd.read_csv("../plots/Standardized_Preprocessing_Plots/model_errors.csv")

rows = []
for model in errors["model"].unique():
    # the reviews this one model got wrong
    texts = errors.loc[errors["model"] == model, "text"].tolist()
    if len(texts) < 30:  # BERTopic needs a reasonable number of reviews to find topics
        print(f"Skipping {model}: only {len(texts)} errors")
        continue

    # smaller topics than usual since each model has few errors; stop_words drops "the", "and", ...
    topic_model = BERTopic(min_topic_size=10, vectorizer_model=CountVectorizer(stop_words="english"))
    topic_model.fit(texts)

    for topic_id in topic_model.get_topics():
        if topic_id == -1:  # -1 is the leftover "no topic" bucket
            continue
        words = [word for word, score in topic_model.get_topic(topic_id)]
        rows.append({"model": model, "topic": topic_id, "topic_words": ", ".join(words)})

pd.DataFrame(rows).to_csv("Model_Topic_Errors.csv", index=False)
