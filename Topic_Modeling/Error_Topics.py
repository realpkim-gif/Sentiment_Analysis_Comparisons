import os
os.environ["USE_TF"] = "0"  # tell transformers to ignore TensorFlow (Keras 3 clash)
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Run from the Topic_Modeling folder.
# Needs model_errors.csv, which GetErrorsAndCorrection.py creates.

errors = pd.read_csv("../plots/Standardized_Preprocessing_Plots/model_errors.csv")

rows = []

#Foorloop for every unique model
for model in errors["model"].unique():
    # the reviews this one model got wrong
    texts = []
    for _, row in errors.iterrows():  # one row at a time for every row, trash index
        if row["model"] == model:  # keep only the rows for this specific model
            texts.append(row["text"])

    if len(texts) < 30:  # BERTopic needs a reasonable number of reviews to find topics
        print(f"Skipping {model}: only {len(texts)} errors which is less than 30")
        continue

    print(f"Finding topics in {model}'s {len(texts)} errors...")

    topic_model = BERTopic(
        vectorizer_model=CountVectorizer(stop_words="english"),
        min_topic_size=50, #sends topics with less than 50 to -1
        nr_topics="auto",
        verbose=True,
    )

    topic_model.fit(texts)

    # topic -1 is the outlier bucket: errors that didn't fit any topic. It's kept here on purpose.
    for topic_id in topic_model.get_topics():
        words = [word for word, score in topic_model.get_topic(topic_id)]
        rows.append({"model": model, "topic": topic_id, "topic_words": ", ".join(words)})

os.makedirs("Model_Topic_Errors", exist_ok=True)
pd.DataFrame(rows).to_csv("Model_Topic_Errors/model_error_topics.csv", index=False)
print("Saved: Model_Topic_Errors/model_error_topics.csv")
