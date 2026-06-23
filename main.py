import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import vader_model, textblob_model, spacy_model, afinn_model, sentiwordnet_model, flair_model, stanza_sentiment, nrc_lexicon

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
flair_result     = flair_model.get_accuracy(df.copy())
print("Running Stanza (already computed on import)...")
stanza_result    = stanza_sentiment.get_accuracy()
print("Running NRC Lexicon (already computed on import)...")
nrc_result       = nrc_lexicon.get_accuracy()

results = [vader_result, textblob_result, afinn_result, spacy_result, swn_result, flair_result, stanza_result, nrc_result]

# --- Accuracy summary dataframe ---
accuracy_df = pd.DataFrame([(r[0], f"{r[1]:.2%}") for r in results], columns=["Model", "Accuracy"])
print("\n--- Model Accuracy Summary ---")
print(accuracy_df)

# --- Per model: prediction distribution plots ---
for name, acc, preds in results:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=preds, order=sorted(set(preds)), palette="Set2", ax=ax)
    ax.set_title(f"{name} — Prediction Distribution (Accuracy: {acc:.2%})")
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/{name.replace(' ', '_').replace('/', '_')}_distribution.png")
    plt.close()
    print(f"Saved: plots/{name}_distribution.png")

# --- Final bar chart: all model accuracies ---
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=accuracy_df, x="Model", y=[r[1] * 100 for r in results], palette="Set2", ax=ax)
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
