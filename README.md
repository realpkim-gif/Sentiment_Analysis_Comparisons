# Sentiment Analysis Comparisons

> Internship project — Kean University, under Dr. Kwak
> Comparing 8 sentiment analysis models on a Twitter dataset using Python.

---

## Objective

To compare the performance of rule-based, lexicon-based, neural network, and machine learning approaches to sentiment analysis on real Twitter data. Each model classifies tweets as **positive**, **negative**, or **neutral**, and is evaluated on accuracy, precision, recall, and F1 score.

---

## Project Overview

This project started as an exploration of different sentiment analysis approaches applied to a Twitter dataset. It began with rule-based models (VADER, TextBlob) to establish a baseline, then expanded to lexicon-based methods (AFINN, SentiWordNet, NRC EmoLex), a neural network model (Flair), a CNN-based model (Stanza), and a machine learning pipeline using spaCy preprocessing with TF-IDF and Logistic Regression.

The project was then organized into a structured Python package with a shared preprocessing module (`preprocessing/clean.py`), individual model files under `models/`, and a `main.py` that imports all models, runs evaluations, and generates comparison visualizations saved to `plots/`.

---

## Models Compared

| Model              | Approach                        | Library / Method                  |
|--------------------|---------------------------------|-----------------------------------|
| VADER              | Rule-based lexicon              | `nltk` (SentimentIntensityAnalyzer) |
| TextBlob           | Rule-based lexicon              | `textblob`                        |
| AFINN              | Weighted lexicon                | `afinn`                           |
| SentiWordNet       | Lexicon + Logistic Regression   | `nltk`, `sklearn`                 |
| NRC EmoLex         | Emotion lexicon                 | Custom lookup + `spacy`           |
| Flair              | Neural network (pretrained)     | `flair`                           |
| Stanza             | CNN-based (pretrained)          | `stanza`                          |
| spaCy + LR         | TF-IDF + Logistic Regression    | `spacy`, `sklearn`                |

---

## Project Structure

```
Sentiment_Chatbot/
├── dataFolder/
│   ├── train.csv                        # Primary dataset (Twitter)
│   ├── sentiment_analysis.csv           # Used by SentiWordNet standalone
│   └── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
├── preprocessing/
│   └── clean.py                         # Shared preprocessing functions
├── models/
│   ├── vader_model.py
│   ├── textblob_model.py
│   ├── afinn_model.py
│   ├── sentiwordnet_sentiment_analysis.py
│   ├── nrc_lexicon.py
│   ├── flair_sentiment_analysis.py
│   ├── stanza_sentiment.py
│   ├── spacy_sentiment_analysis.py
│   └── *.ipynb                          # Original notebooks kept as reference
├── notes/                               # Personal analysis notes per model
├── plots/                               # Output charts (auto-generated)
├── main.py                              # Runs all models and saves visualizations
└── README.md
```

---

## Requirements

- **Python:** 3.13+
- **Hardware:** A reasonably modern CPU is sufficient; a GPU is recommended for Flair and Stanza due to significantly slower inference on CPU.

**Key libraries:**
```
pandas, numpy, matplotlib, seaborn, sklearn,
nltk, textblob, afinn, spacy, flair, stanza
```

Install spaCy's English model:
```bash
python -m spacy download en_core_web_sm
```

Download required NLTK data (run once):
```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

---

## How to Run

Ensure the working directory is the **project root** (`Sentiment_Chatbot/`) so that relative paths like `dataFolder/train.csv` resolve correctly.

```bash
python main.py
```

This will:
1. Load `dataFolder/train.csv`
2. Run all 8 models sequentially
3. Print an accuracy summary table to the console
4. Save per-model prediction distribution plots and an overall accuracy comparison chart to `plots/`

> **Note:** Flair and Stanza process the full dataset and may take several minutes depending on hardware.

---

## Outcomes

| Model          | Accuracy |
|----------------|----------|
| VADER          | *(to be filled)* |
| TextBlob       | *(to be filled)* |
| AFINN          | *(to be filled)* |
| SentiWordNet   | *(to be filled)* |
| NRC EmoLex     | *(to be filled)* |
| Flair          | *(to be filled)* |
| Stanza         | *(to be filled)* |
| spaCy + LR     | *(to be filled)* |

---

## Limitations

- **Dataset bias:** All models were evaluated on Twitter data. Models trained on formal text (Stanza, Flair) underperform on slang, abbreviations, and informal language common in tweets.
- **NRC EmoLex** is an emotion lexicon, not a true sentiment classifier. It was included for comparison but is not well-suited for 3-class sentiment prediction.
- **AFINN** has a small lexicon and performs poorly on out-of-vocabulary social media language.
- **Runtime:** Flair and Stanza are significantly slower than other models due to neural inference on the full dataset.
- **SentiWordNet standalone** uses a different CSV (`sentiment_analysis.csv`) than what `main.py` passes in (`train.csv`). Results between standalone and `main.py` runs may differ slightly.

---

## Warnings

- The `.py` files inside `models/` that were converted from Jupyter notebooks (`sentiwordnet_sentiment_analysis.py`, `flair_sentiment_analysis.py`, `spacy_sentiment_analysis.py`) may be missing some inline comments and markdown section headers present in the original `.ipynb` files. The original notebooks are kept alongside each `.py` file for reference.
- Stanza and NRC EmoLex run all preprocessing on import, which means importing them in `main.py` triggers full dataset processing immediately.

---

## Simple Summary

| Model        | Label                            | Key Characteristics                                                                 |
|--------------|----------------------------------|-------------------------------------------------------------------------------------|
| VADER        | Most Used for Social Media       | Lexicon-based, rule-based scoring, designed for social media                        |
| TextBlob     | Beginner Friendly                | Lexicon-based, simple polarity scoring, good for clean text, can also predict subjectivity (0 = objective, 1 = subjective) |
| AFINN        | Simplest Lexicon                 | Weighted word scores, small vocabulary                                              |
| SpaCy + LR   | Most Accurate (in this project)  | ML-based, TF-IDF + Logistic Regression, spaCy tokenizer                            |
| SentiWordNet | Most Linguistically Detailed     | WordNet POS lookup, scores nouns, adjectives, and adverbs                           |
| Flair        | Best Neural Network for Reviews  | Pretrained neural network, outputs a confidence score per prediction, binary only (positive/negative) |
| Stanza       | Best for Formal Text             | CNN-based, trained on formal text, multilingual, high accuracy on news/books        |
| NRC Lexicon  | Best for Emotion Analysis        | Maps words to 8 emotions + pos/neg, multilingual, captures complex emotional signal |

---

## Model Capabilities

| Model        | POS Tagging | Subjectivity | Notes                                                                 |
|--------------|-------------|--------------|-----------------------------------------------------------------------|
| VADER        | No          | No           | Pure rule-based heuristics — no grammatical analysis whatsoever       |
| TextBlob     | Yes         | Yes          | POS tagging available but not used in sentiment scoring; subjectivity score (0 = objective, 1 = subjective) is a separate output |
| AFINN        | No          | No           | Simple weighted word lookup only                                      |
| SentiWordNet | Yes         | No           | POS tagging is central to scoring — same word scored differently as noun vs verb |
| NRC Lexicon  | No          | No           | Emotion lookup only                                                   |
| Flair        | No          | No           | Neural network — binary only (positive/negative), outputs confidence score per prediction |
| Stanza       | Yes         | No           | Full NLP pipeline including POS, NER, dependency parsing              |
| SpaCy + LR   | Yes         | No           | spaCy used for tokenization and lemmatization before TF-IDF           |

---

## Conclusion

---

## Notes

- AI assistance (Claude) was used for project organization and code structuring — specifically refactoring notebook code into importable modules (`preprocessing/clean.py`, `models/`) and organizing visualization code in `main.py`. All model logic, analysis, interpretation of results, and implementation decisions were completed and reviewed independently.
- Some comments from the original Jupyter notebooks may be missing in the converted `.py` files. Refer to the `.ipynb` files in `models/` for the full annotated versions.
