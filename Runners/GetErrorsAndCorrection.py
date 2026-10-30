#All models get standardized_preprocessing

import os
import pandas as pd
import matplotlib
from Topic_Modeling.Berttopic import *
matplotlib.use("Agg")  # only save plots to files, don't open a window (no Qt needed)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models import vader_model, textblob_model, afinn_model, stanza_sentiment, nrc_lexicon
from preprocessing.clean import preprocess
from models import spacy_sentiment_analysis as spacy_model
from models import sentiwordnet_sentiment_analysis as sentiwordnet_model
from models import flair_sentiment_analysis as flair_model



os.makedirs("../plots/Standardized_Preprocessing_Plots", exist_ok=True)

df = pd.read_csv(r"../dataFolder/train.csv", encoding="latin-1")
df = df[["text", "sentiment"]].fillna("")
df = preprocess(df)

# --- Run all models ---
def run(name, predict, data):
    print(f"Running {name}...")
    preds = predict(data.copy())
    true = df.loc[preds.index, "sentiment"]
    return (name, accuracy_score(true, preds), list(preds)), true

df_binary = df[df["sentiment"] != "neutral"].copy()

runs = [run("VADER", vader_model.predict, df),
        run("TextBlob", textblob_model.predict, df),
        run("AFINN", afinn_model.predict, df),
        run("SpaCy + LR", spacy_model.predict, df),
        run("SentiWordNet", sentiwordnet_model.predict, df),
        run("Flair", flair_model.predict, df_binary),
        run("Stanza", stanza_sentiment.predict, df),
        run("NRC Lexicon", nrc_lexicon.predict, df)]

results = [r[0] for r in runs]       # each is (model name, accuracy, list of predictions)
true_labels = [r[1] for r in runs]   # the correct labels for each model, in the same order

def scoring(inference, label):
    f1        = f1_score(label, inference, average="weighted", zero_division=0)
    precision = precision_score(label, inference, average="weighted", zero_division=0)
    recall    = recall_score(label, inference, average="weighted", zero_division=0)
    return f1, precision, recall


def saveErrors():
    errors = []
    for i in range(len(results)):
        name, acc, preds = results[i]
        true = true_labels[i]
        for j in range(len(preds)):
            if preds[j] != true.iloc[j]:  # wrong prediction
                review_id = true.index[j]  # row number of this review in df
                errors.append({"model": name,
                               "text": df.loc[review_id, "text"],
                               "true": true.iloc[j],
                               "predicted": preds[j]})

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv("../plots/Standardized_Preprocessing_Plots/model_errors.csv", index=False)
    print(f"Saved {len(errors_df)} errors: ../plots/Standardized_Preprocessing_Plots/model_errors.csv")
    return errors_df

def saveCorrect():
    correct=[]
    for i in range(len(results)):
        name, acc, preds = results[i]
        true = true_labels[i]
        for j in range(len(preds)):
            if preds[j] == true.iloc[j]:
                correct.append({
                    "model": name,
                    "text": df.loc[j, "text"],
                    "true": true.iloc[j],
                    "predicted": preds[j] #predicted and true should be save
                })
    correct_df = pd.DataFrame(correct)
    correct_df.to_csv("../plots/Standardized_Preprocessing_Plots/model_correct.csv", index=False)
    print(f"Saved {len(correct_df)} errors: ../plots/Standardized_Preprocessing_Plots/model_correct.csv")
    return correct_df

saveErrors()
saveCorrect()

# Load missed topics into Topic_Modeling
df_loaded_false = pd.read_csv("../plots/Standardized_Preprocessing_Plots/model_errors.csv")
models_topic_error = pd.DataFrame(columns=["model", "topic_words"])

for i in df['model'].unique():
    for j in BertTopic.topic_model.get_topics():
        models_topic_error[len(df_loaded_false), "topic_words"] = BertTopic(df_loaded_false["text"][j])

models_topic_error.to_csv("Topic_Modeling/Model_Topic_Errors", index=False)
