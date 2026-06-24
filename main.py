import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from models import vader_model, textblob_model, afinn_model, stanza_sentiment, nrc_lexicon
from models import spacy_sentiment_analysis as spacy_model
from models import sentiwordnet_sentiment_analysis as sentiwordnet_model
from models import flair_sentiment_analysis as flair_model
from sklearn.metrics import confusion_matrix

os.makedirs("plots", exist_ok=True)

df = pd.read_csv(r"dataFolder/train.csv", encoding="latin-1")
df = df[["text", "sentiment"]].fillna("")

# --- Run all models ---
print("Running VADER...")
vader_result     = vader_model.get_accuracy(df.copy())
print("Running TextBlob...")
textblob_result  = textblob_model.get_accuracy(df.copy())
print("Running AFINN...")
afinn_result     = afinn_model.get_accuracy(df.copy())
print("Running SpaCy + LR...")
spacy_result     = spacy_model.get_accuracy(df.copy())
print("Running SentiWordNet...")
swn_result       = sentiwordnet_model.get_accuracy(df.copy())

print("Running Flair...")
df_binary = df[df["sentiment"] != "neutral"].copy()
flair_result     = flair_model.get_accuracy(df_binary)

print("Running Stanza (already computed on import)...")
stanza_result    = stanza_sentiment.get_accuracy()
print("Running NRC Lexicon (already computed on import)...")
nrc_result       = nrc_lexicon.get_accuracy()

results = [vader_result, textblob_result, afinn_result, spacy_result, swn_result, flair_result, stanza_result, nrc_result]
true_labels = [df["sentiment"], df["sentiment"], df["sentiment"], df["sentiment"], df["sentiment"],
               df_binary["sentiment"], df["sentiment"], df["sentiment"]]

def scoring(inference, label):
    f1        = f1_score(label, inference, average="weighted", zero_division=0)
    precision = precision_score(label, inference, average="weighted", zero_division=0)
    recall    = recall_score(label, inference, average="weighted", zero_division=0)
    return f1, precision, recall



# --- Scoring summary ---
rows = []
for i in range(len(results)):
    name, acc, preds = results[i]
    true = list(true_labels[i])
    f1, precision, recall = scoring(preds, true)
    print(f"{name}: Accuracy={acc:.2%}  F1={f1:.2%}  Precision={precision:.2%}  Recall={recall:.2%}")
    rows.append({"Model": name, "Accuracy": f"{acc:.2%}", "F1": f"{f1:.2%}", "Precision": f"{precision:.2%}", "Recall": f"{recall:.2%}"})

scores_df = pd.DataFrame(rows)
print("\n--- Model Scores Summary ---")
print(scores_df)
scores_df.to_csv("model_scores.csv", index=False)
print("Saved: model_scores.csv")


confusion = []
for i in range(len(results)):
    print(f"Confusion Matrix for {results[i][0]}")
    confusion.append(confusion_matrix(list(true_labels[i]), results[i][2]))

for i, cm in enumerate(confusion):
    labels = sorted(set(true_labels[i]) | set(results[i][2]))
    name = results[i][0]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"plots/{name}_confusion_matrix.png", bbox_inches="tight")
    plt.close()




# --- Per model: prediction distribution plots ---
for i in range(len(results)):
    name, acc, preds = results[i]
    f1  = scores_df.loc[scores_df["Model"] == name, "F1"].values[0]
    pre = scores_df.loc[scores_df["Model"] == name, "Precision"].values[0]
    rec = scores_df.loc[scores_df["Model"] == name, "Recall"].values[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=preds, hue=preds, order=sorted(set(preds)), palette="Set2", ax=ax, legend=False)
    ax.set_title(f"{name} — Prediction Distribution")
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Count")
    plt.figtext(0.5, -0.04, f"Accuracy: {acc:.2%}  |  F1: {f1}  |  Precision: {pre}  |  Recall: {rec}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"plots/{name.replace(' ', '_').replace('/', '_')}_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: plots/{name}_distribution.png")

# --- Final bar chart: all model accuracies ---
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=scores_df["Model"], y=[r[1] * 100 for r in results], palette="Set2", ax=ax)
ax.set_title("Model Accuracy Comparison")
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
for i, (_, acc, _) in enumerate(results):
    ax.text(i, acc * 100 + 1, f"{acc:.2%}", ha="center", fontsize=9)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plots/all_models_accuracy.png")
plt.close()
print("Saved: plots/all_models_accuracy.png")
