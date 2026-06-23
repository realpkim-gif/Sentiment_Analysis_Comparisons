# NRC Word-Emotion Association Lexicon or EmoLex is only a applicable lexicon, not a sentiment analyzer model
import spacy
import pandas as pd
from spacy.cli import download
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Preprocess with spaCy
def preprocess(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove contractions apostrophes
    text = re.sub(r'`\w+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = text.strip()
    #remove elongated words and make them into normal words ex: sooooo -> so
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    doc = nlp(text)  # technically does all the preprocessing in this line with spaCy
    tokens = [
        token.lemma_.lower()     # lemmatize: "running" → "run": EmoLex has run but not running. Lemmatization maximizes lookup hits.
        for token in doc
        if not token.is_stop     # not in stopwords
        and not token.is_punct   # not punctuation
        and not token.is_space   # not whitespace
    ]
    return " ".join(tokens)      # return string not list — easier to save/load from CSV

df = pd.read_csv(r"dataFolder/train.csv", encoding="latin-1")
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]

print("Starting preprocess...")
df["text"] = df["text"].apply(preprocess)
df.to_csv("data_spaCy.csv", index=False)


#using lexicon

lexicon = pd.read_csv("dataFolder/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                      names=["word", "emotion", "association"],
                      sep='\t')

lexicon = lexicon[lexicon["association"] == 1]

# Build lookup dictionary {word: [list of emotions]}
emolex = lexicon.groupby("word")["emotion"].apply(list).to_dict()

# ── 3. Replace words with their emotions ─────────
def replace_with_emotions(text):
    result = []
    for token in text.split():        # split string into words
        if token in emolex:
            result.extend(emolex[token])
        else:
            pass

    return " ".join(result)

# ── 4. Apply to every row ────────────────────────
print("Replacing words with emotions...")
df["emotion_text"] = df["text"].apply(replace_with_emotions)



def classify(emotion_text):
    words = emotion_text.split()

    # Count every emotion
    scores = {
        "anger": words.count("anger"),
        "fear": words.count("fear"),
        "anticipation": words.count("anticipation"),
        "trust": words.count("trust"),
        "surprise": words.count("surprise"),
        "sadness": words.count("sadness"),
        "joy": words.count("joy"),
        "disgust": words.count("disgust"),
        "positive": words.count("positive"),
        "negative": words.count("negative")
    }

    emotion_only = {k: v for k, v in scores.items()
                    if k not in ["positive", "negative"]}

    dominant_emotion = max(emotion_only, key=lambda x: emotion_only[x])

    if emotion_only[dominant_emotion] == 0:
        dominant_emotion = "neutral"

    # ── Positive / Negative rule ─────────────────
    negative_emotions = ["anger", "fear", "sadness", "disgust", "negative"]
    positive_emotions = ["joy", "trust", "anticipation", "surprise", "positive"]

    pos_score = sum(scores[e] for e in positive_emotions)
    neg_score = sum(scores[e] for e in negative_emotions)

    if pos_score > neg_score:
        sentiment = "positive"
    elif neg_score > pos_score:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment, dominant_emotion


# Apply — returns two columns
print("classifying text...")
df["sentiment_predicted"], df["dominant_emotion"] = zip(*df["emotion_text"].apply(classify)) # zip(*) splits a list of tuples into separate lists — zip pairs them together

print(df[["text", "sentiment_predicted","emotion_text"]].head(10))

def evaluate(df):
    correct = 0
    wrong = 0
    total = 0
    emotion_series = []

    for i in range(len(df)):
        true     = df["sentiment"].iloc[i].lower()
        pred     = df["sentiment_predicted"].iloc[i].lower() if df["sentiment_predicted"].iloc[i] else ""
        dominant = df["dominant_emotion"].iloc[i]

        if true == pred:
            correct += 1
        else:
            wrong += 1
        total += 1

        emotion_series.append({
            "text":             df["text"].iloc[i],
            "true_sentiment":   true,
            "pred_sentiment":   pred,
            "dominant_emotion": dominant,
            "correct":          true == pred
        })

    # Align labels for sklearn — make sure formats match
    true_labels = df["sentiment"].str.lower()
    pred_labels = df["sentiment_predicted"].str.lower()

    # Accuracy stats
    print(f"Correct : {correct}/{total}")
    print(f"Accuracy: {accuracy_score(true_labels, pred_labels, zero_division=0):.2%}")
    print(f"Precision: {precision_score(true_labels, pred_labels, average='weighted', zero_division=0):.2%}")
    print(f"Recall:    {recall_score(true_labels, pred_labels, average='weighted', zero_division=0):.2%}")
    print(f"F1 Score:  {f1_score(true_labels, pred_labels, average='weighted', zero_division=0):.2%}")

    # Dominant emotion distribution
    emotion_df = pd.DataFrame(emotion_series)
    print("\nDominant Emotion Distribution:")
    print(emotion_df["dominant_emotion"].value_counts())

    print("\nAccuracy per Emotion:")
    print(emotion_df.groupby("dominant_emotion")["correct"]
          .mean()
          .sort_values(ascending=False)
          .map("{:.2%}".format))

    # Save to CSV
    emotion_df.to_csv("emotion_series.csv", index=False)
    print("\nSaved line by line results to emotion_series.csv")

    return emotion_df

emotion_series = evaluate(df)

def get_accuracy():
    from sklearn.metrics import accuracy_score
    true = df["sentiment"].str.lower()
    pred = df["sentiment_predicted"].str.lower()
    acc = accuracy_score(true, pred)
    preds = list(pred)
    return ("NRC Lexicon", acc, preds)



"""
[10 rows x 3 columns]
Correct : 14158/27480
Wrong   : 13322/27480
Accuracy: 51.52%
Precision: 53.29%
Recall:    51.52%
F1 Score:  50.82%

Dominant Emotion Distribution:
dominant_emotion
neutral         9323
anticipation    6905
trust           2959
anger           2867
fear            1851
joy             1603
sadness          958
surprise         745
disgust          269
Name: count, dtype: int64

Accuracy per Emotion:
dominant_emotion
joy             69.49%
disgust         60.97%
sadness         59.81%
anger           56.64%
neutral         52.83%
fear            51.54%
anticipation    50.96%
surprise        36.11%
trust           34.34%
Name: correct, dtype: object

Saved line by line results to emotion_series.csv




Ultimately, NRC is not very good. The only practical use would be for the ability to get multiple emotions and get the most apparent emotion out of a text.
If thats not the case, just use Vader if you reall y want to go with a lexicon approach. If not use Stanza or Flair.
"""