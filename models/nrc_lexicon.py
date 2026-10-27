# NRC Word-Emotion Association Lexicon or EmoLex is only a applicable lexicon, not a sentiment analyzer model
import os
import pandas as pd

# lemmatized text (lemma_text from preprocess) maximizes lookup hits: "running" → "run": EmoLex has run but not running

#using lexicon

lexicon = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataFolder", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"),
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


def predict(df):
    emotion_text = df["lemma_text"].apply(replace_with_emotions)
    sentiment, dominant_emotion = zip(*emotion_text.apply(classify)) # zip(*) splits a list of tuples into separate lists — zip pairs them together
    return pd.Series(sentiment, index=df.index).str.lower()



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