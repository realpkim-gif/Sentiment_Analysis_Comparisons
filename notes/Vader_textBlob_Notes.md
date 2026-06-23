# Sentiment Analysis — VADER & TextBlob Complete Notes

---

## Table of Contents
1. [Preprocessing — What to do and what NOT to do](#preprocessing)
2. [VADER — How it works](#vader)
3. [TextBlob — How it works](#textblob)
4. [Side by Side Comparison](#comparison)
5. [Common Bugs & Fixes](#bugs)
6. [Evaluation](#evaluation)
7. [Limitations & Next Steps](#limitations)

---

## 1. Preprocessing <a name="preprocessing"></a>

### What TO clean (helps both tools)
- HTML tags and entities (`&amp;`, `<p>`, `<br>`)
- URLs
- Encoding errors (byte 0xb2 etc.)

### What NOT to clean (hurts both tools)
| Step | VADER | TextBlob |
|---|---|---|
| Remove punctuation | ❌ Never | ⚠️ Avoid |
| Lemmatize / stem | ❌ Never | ❌ Never |
| Lowercase | ❌ Avoid | ⚠️ Usually OK |
| Remove stopwords | ❌ Never | ❌ Never |
| Remove URLs/HTML | ✅ Yes | ✅ Yes |

> VADER uses punctuation (`!!!`), capitalization (`GREAT`), and stopwords (`not`, `never`) as sentiment signals.
> Removing them actively destroys accuracy.

### Cleaning code
```python
from bs4 import BeautifulSoup

# Strip HTML (also decodes &amp; &nbsp; etc.)
def strip_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

df["text_clean"] = df["text"].apply(strip_html)

# Fix encoding when reading file
df = pd.read_csv("file.csv", encoding="latin-1")  # fixes byte 0xb2 errors
```

---

## 2. VADER <a name="vader"></a>

### Background
- Created by **C.J. Hutto & Eric Gilbert (2014)**
- Built specifically for **social media text**
- Lexicon crowd-sourced via Amazon Mechanical Turk
- Workers rated ~7500 words/emojis/slang on scale of **-4 to +4**

### Step 1 — Lexicon Lookup
VADER has ~7500 tokens with pre-rated scores:

| Token | Score |
|---|---|
| "good" | 1.9 |
| "great" | 3.1 |
| "terrible" | -3.4 |
| "ok" | 0.9 |
| "💔" | -2.2 |
| "😊" | 2.1 |

Words **not in the lexicon score 0.0** — this is why conversational/indirect text fails.

### Step 2 — The 5 Heuristic Rules

**Rule 1 — Punctuation**
Each `!` adds `0.292` to the absolute score (capped at 3):
```
"great"    → 1.9
"great!"   → 1.9 + 0.292 = 2.192
"great!!"  → 1.9 + 0.584 = 2.484
"great!!!" → 1.9 + 0.876 = 2.776  (max)
```

**Rule 2 — Capitalization**
ALL CAPS adds `0.733` to the absolute score:
```
"great"  → 1.9
"GREAT"  → 1.9 + 0.733 = 2.633
```

**Rule 3 — Degree Modifiers (Boosters)**
Words before a sentiment word boost or dampen it:
```
"good"           → 1.9
"extremely good" → 1.9 * 1.293 = 2.457   (boost)
"barely good"    → 1.9 * 0.293 = 0.557   (damp)
```
Booster words: "very", "extremely", "incredibly", "barely", "somewhat", "kind of"

**Rule 4 — Negation**
Negation word within 3 tokens multiplies score by `-0.74`:
```
"good"     →  1.9
"not good" →  1.9 * -0.74 = -1.406

"not bad"  → -3.4 * -0.74 = +2.516  (correctly positive)
```
Negation words: "not", "never", "no", "nothing", "neither", "hardly", "can't", "won't", "don't"

**Rule 5 — "But" Conjunction**
Sentiment **after "but"** is weighted more heavily than before:
```
"The food was great but the service was terrible"
→ "great" score gets reduced
→ "terrible" score gets amplified
```

### Step 3 — Compound Score Formula

**1. Sum all modified token scores:**
```
sum_s = score_1 + score_2 + ... + score_n
```

**2. Normalize using:**
```
compound = sum_s / sqrt(sum_s² + alpha)

where alpha = 15  (smoothing constant)
```
This squishes the result to between **-1.0 and +1.0**

**Full example** — `"GREAT food, not bad service!"`:
```
GREAT    → 1.9 + 0.733 (caps)   = 2.633
not bad  → -3.4 * -0.74         = 2.516
!        →                       + 0.292

sum_s = 2.633 + 2.516 + 0.292  = 5.441

compound = 5.441 / sqrt(5.441² + 15)
         = 5.441 / sqrt(29.6 + 15)
         = 5.441 / sqrt(44.6)
         = 5.441 / 6.678
         = 0.815  → very positive ✅
```

**3. Apply threshold:**
```
compound >= 0.05  → positive
compound <= -0.05 → negative
else              → neutral
```

### Why texts score 0.0
Any text with no lexicon words returns exactly 0.0:
```
"I'd have responded if I were going"  → no sentiment words → 0.0
"Sons of ****"                        → masked word        → 0.0
"Both of you"                         → no sentiment words → 0.0
"Uh oh, I am sunburned"               → sunburned unknown  → 0.0
```

### VADER Code
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create ONCE outside the function — not inside the loop
vader = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = vader.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["vader_pred"] = df["text"].apply(vader_sentiment)

# Check compound score distribution
df["compound"] = df["text"].apply(lambda x: vader.polarity_scores(x)["compound"])
print(df["compound"].value_counts().head(10))
```

> ⚠️ Never create `SentimentIntensityAnalyzer()` inside the loop — it reloads the lexicon every row and makes it ~100x slower.

---

## 3. TextBlob <a name="textblob"></a>

### Background
- Built on top of **Pattern** (University of Antwerp)
- Lexicon built from **movie reviews, product reviews, general web text**
- Broader coverage than VADER but fewer social media features
- Uses **POS tagging** — understands grammar, not just keywords

### Step 1 — Lexicon
Each word has **3 values**:

| Word | Polarity | Subjectivity |
|---|---|---|
| "good" | 0.7 | 0.6 |
| "terrible" | -1.0 | 1.0 |
| "ok" | 0.2 | 0.4 |
| "beautiful" | 1.0 | 1.0 |
| "allegedly" | 0.0 | 0.1 |
| "murder" | -1.0 | 0.1 |
| "the" | 0.0 | 0.0 |

- **Polarity**: -1.0 (negative) → +1.0 (positive)
- **Subjectivity**: 0.0 (objective/fact) → 1.0 (subjective/opinion)

### Step 2 — POS Tagging
TextBlob runs Part of Speech tagging first:
```
"The food was absolutely terrible"
 DT   NN   VBD  RB          JJ

DT  = determiner
NN  = noun
VBD = verb past tense
RB  = adverb  ← modifier
JJ  = adjective ← primary sentiment carrier
```
Looks for: **adjectives (JJ)**, **adverbs (RB)**, **sentiment verbs (VB)**

This is why TextBlob handles complete sentences better than VADER.

### Step 3 — Modifier Handling
```
"absolutely terrible"
→ "absolutely" = intensifier → multiplier 1.3
→ "terrible"   = -1.0
→ modified      = -1.0 * 1.3 = -1.3  (capped at -1.0)

"not good"
→ "not"  = negation → flip polarity
→ "good" = 0.7
→ result = 0.7 * -1.0 = -0.7
```

### Step 4 — Averaging (NOT summing like VADER)
TextBlob **averages** scores across all sentiment words:
```
"The food was great and the service was terrible"

great    → polarity =  0.8
terrible → polarity = -1.0

polarity     = (0.8 + -1.0) / 2 = -0.1   → slightly negative
subjectivity = (0.75 + 1.0) / 2 = 0.875  → very subjective
```

### Subjectivity Calculation
Subjectivity is averaged the same way as polarity:
```
"The movie was beautiful but boring"
beautiful → subjectivity = 1.0
boring    → subjectivity = 1.0
result    = (1.0 + 1.0) / 2 = 1.0  → pure opinion

"The president signed the bill on Tuesday"
all words → subjectivity = 0.0
result    = 0.0  → pure fact

"The president allegedly made a terrible decision"
allegedly → 0.1
terrible  → 1.0
result    = (0.1 + 1.0) / 2 = 0.55  → mixed
```

### Using Subjectivity as a Filter
```python
def textblob_sentiment(text):
    blob = TextBlob(text)

    # Objective sentences are likely neutral — don't force a label
    if blob.sentiment.subjectivity < 0.3:
        return "neutral"

    score = blob.sentiment.polarity
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"
```

### TextBlob Code
```python
from textblob import TextBlob

def textblob_sentiment(text):
    score = TextBlob(text).sentiment.polarity

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["textblob_pred"] = df["text"].apply(textblob_sentiment)
```

> ⚠️ Never use `' '.join(text)` on a string — it joins individual characters.  
> ⚠️ Always extract `.sentiment.polarity` — you can't compare a TextBlob object to a number directly.

---

## 4. Side by Side Comparison <a name="comparison"></a>

| | VADER | TextBlob |
|---|---|---|
| **Lexicon size** | ~7,500 | ~2,800 (broader coverage) |
| **Built for** | Social media | General/formal text |
| **Uses POS tagging** | ❌ No | ✅ Yes |
| **Handles ALL CAPS** | ✅ Yes | ❌ No |
| **Handles emojis** | ✅ Yes | ❌ No |
| **Handles `!!!`** | ✅ Yes | ❌ No |
| **Negation** | ✅ Within 3 tokens | ✅ Pattern-based |
| **Output** | pos/neg/neu + compound | polarity + subjectivity |
| **Math** | Normalization formula | Simple average |
| **Speed** | Slower | Much faster |
| **Unknown words** | Score 0.0 | Score 0.0 |
| **Best for** | Tweets, reviews, emojis | Sentences, articles, conversation |

### When TextBlob wins over VADER
- Conversational/indirect language
- Complete grammatical sentences
- Texts with masked profanity (`****`)
- Datasets with subtle sentiment ("my boss is bullying me")
- Longer documents

### When VADER wins over TextBlob
- Raw social media with emojis
- Heavy use of `!!!` and ALL CAPS
- Slangy/informal keyword-heavy text
- Short punchy sentences ("LOVE this!!!" / "worst day ever 😭")

> **Rule:** Never assume — always test both on your actual data and compare.

---

## 5. Common Bugs & Fixes <a name="bugs"></a>

### Label mismatch
```python
# VADER outputs "pos"/"neg"/"neu"
# Dataset uses "positive"/"negative"/"neutral"
# Fix — use matching labels in your function:

def vader_sentiment(text):
    score = vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"   # not "pos"
    elif score <= -0.05:
        return "negative"   # not "neg"
    else:
        return "neutral"    # not "neu"
```

### Wrong syntax in lambda
```python
# BROKEN
df["vader_pred"] = df["text"].apply(lambda x:
    "pos" if vader.polarity_scores(x)["compound"] >= 0.05,  # comma = error
    if vader.polarity_scores(x)["compound"] <= -0.05,       # double if = error
    else "neu"
)

# FIXED
df["vader_pred"] = df["text"].apply(lambda x:
    "positive" if vader.polarity_scores(x)["compound"] >= 0.05
    else "negative" if vader.polarity_scores(x)["compound"] <= -0.05
    else "neutral"
)
```

### Creating analyzer inside loop (SLOW)
```python
# SLOW — reloads lexicon every row
df["vader_pred"] = df["text"].apply(lambda x:
    SentimentIntensityAnalyzer().polarity_scores(x)["compound"]
)

# FAST — create once outside
vader = SentimentIntensityAnalyzer()
df["vader_pred"] = df["text"].apply(lambda x:
    vader.polarity_scores(x)["compound"]
)
```

### Wrong join on string
```python
# BROKEN — joins characters not words
TextBlob(' '.join("hello world"))  # → "h e l l o   w o r l d"

# FIXED — pass string directly
TextBlob("hello world")
```

### NaN values crashing apply
```python
# Check for nulls
print(df["text"].isna().sum())

# Fix
df["text"] = df["text"].fillna("")
```

### Wrong bracket on .loc
```python
# BROKEN
df.loc(df["text"].isna())   # () = TypeError

# FIXED
df.loc[df["text"].isna()]   # [] = correct
df[df["text"].isna()]       # also correct
```

### Encoding error on read
```python
# UnicodeDecodeError: can't decode byte 0xb2
df = pd.read_csv("file.csv", encoding="latin-1")   # fixes 90% of cases
df = pd.read_csv("file.csv", encoding="cp1252")    # Windows fallback
df = pd.read_csv("file.csv", encoding="utf-8", errors="ignore")  # nuclear option
```

---

## 6. Evaluation <a name="evaluation"></a>

### Simple accuracy function
```python
def accuracy(true_labels, predicted_labels):
    correct = 0

    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            correct += 1

    total = len(true_labels)
    accuracy = correct / total

    return accuracy

print(accuracy(df["sentiment"], df["textblob_pred"]))
print(accuracy(df["sentiment"], df["vader_pred"]))
```

### Per-class breakdown
```python
def count(model):
    for label in df["sentiment"].unique():
        correct = sum((df["sentiment"] == label) & (df[model] == label))
        total = sum(df["sentiment"] == label)
        print(f"{label} — {correct}/{total}")

count("vader_pred")
count("textblob_pred")
```

### Full evaluation with sklearn
```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(true_labels, predicted_labels, model_name="Model"):
    accuracy  = accuracy_score(true_labels, predicted_labels)
    f1        = f1_score(true_labels, predicted_labels, average="weighted")
    precision = precision_score(true_labels, predicted_labels, average="weighted")
    recall    = recall_score(true_labels, predicted_labels, average="weighted")

    print(f"--- {model_name} ---")
    print(f"Accuracy  : {accuracy:.2%}")
    print(f"F1 Score  : {f1:.2%}")
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print()

evaluate_model(df["sentiment"], df["textblob_pred"], "TextBlob")
evaluate_model(df["sentiment"], df["vader_pred"],    "VADER")
```

### Why accuracy alone is misleading
If VADER predicts everything as neutral and neutral is 40% of your data:
- Accuracy = 40% (looks ok)
- F1 on positive/negative = 0.0 (completely useless)

**Always check per-class breakdown AND F1 score, not just accuracy.**

### VADER all-neutral problem
```python
# Diagnose — check score distribution
df["compound"] = df["text"].apply(lambda x: vader.polarity_scores(x)["compound"])
print(df["compound"].value_counts().head(10))

# If most scores are 0.0, VADER can't see sentiment in your text
# Option 1 — use TextBlob as fallback when score is 0.0
def combined_sentiment(text):
    score = vader.polarity_scores(text)["compound"]

    if score == 0.0:
        return textblob_sentiment(text)  # fallback
    elif score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"
```

---

## 7. Limitations & Next Steps <a name="limitations"></a>

### What neither tool can handle
```
"my boss is bullying me"          → should be negative, scores neutral
"leave me alone"                  → should be negative, scores neutral
"I'd have responded if I were going" → neutral, correct but for wrong reason
"Sons of ****"                    → masked word, scores 0.0
"Soooo high"                      → ambiguous, scores wrong
```

Both tools are **lexicon-based** — they look up words in a dictionary.
They have zero understanding of context, sarcasm, or indirect sentiment.

### Label noise
If your dataset labels are inconsistent, no tool can hit 100%:
```
"I feel sorry, I miss you" → labeled positive, clearly negative language
"Don't angry me"           → labeled negative, ambiguous
```
This creates a ceiling on accuracy that preprocessing cannot fix.

### The real next step — Transformers
```python
pip install transformers

from transformers import pipeline

# RoBERTa trained on 58M tweets — understands context
model = pipeline("sentiment-analysis",
                 model="cardiffnlp/twitter-roberta-base-sentiment")

result = model("my boss is bullying me")
# → correctly negative ✅
```

Transformers read the **whole sentence as context** instead of word by word.
That's the jump needed to handle indirect, conversational, and sarcastic text.