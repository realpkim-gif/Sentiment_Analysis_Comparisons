# Stanza — sentiment analysis notes

## What is Stanza
- NLP library from Stanford NLP Group
- Fully neural — every component is a trained neural network, no rule-based fallbacks
- Multilingual — 70+ languages using the same pipeline interface
- Built on PyTorch

## Pipeline architecture
Each processor is a separate neural model chained in sequence:
```
tokenize → mwt → pos → lemma → depparse → ner → sentiment

```
You can load the full chain or just the processors you need.

## Setup

```python
pip
install
stanza

from models import stanza_sentiment

stanza.download('en')  # ~500MB pretrained English models
nlp = stanza.Pipeline('en')  # loads full pipeline
```

Selective loading (faster):
```python
nlp = stanza.Pipeline('en', processors='tokenize,sentiment')
```

## Running sentiment analysis
```python
doc = nlp("The movie was not bad at all.")

for sent in doc.sentences:
    print(sent.text, sent.sentiment)
# sentiment: 0 = negative, 1 = neutral, 2 = positive
```

## The sentiment model
- Trained on **Stanford Sentiment Treebank (SST)** — ~11k human-labeled movie review sentences
- Labels applied at every node of the parse tree, not just the full sentence
- Two variants:
  - **CNN** — convolutional filters (n-gram kernels) over word embeddings → max pool → feedforward → softmax
  - **Tree-LSTM** — computation follows the constituency parse tree; sentiment propagates bottom-up through grammatical structure
- Word vectors: pretrained GloVe or fastText embeddings (not a hand-crafted lexicon)
- Output: 3 classes per sentence — negative / neutral / positive

## Why Tree-LSTM beats a lexicon
A lexicon scores "not bad" as negative (two negative words). Tree-LSTM understands that "not bad" is a constituent where negation flips polarity — because it reasons over parse tree structure, not left-to-right word order.

## Navigating the output object
```python
doc                         # whole document
doc.sentences               # list of sentences
doc.sentences[0].sentiment  # 0 / 1 / 2
doc.sentences[0].tokens     # list of tokens
doc.sentences[0].words      # list of words (usually 1 per token)

word = doc.sentences[0].words[0]
word.text      # surface form
word.upos      # universal POS tag (NOUN, VERB, ADJ...)
word.lemma     # base form
word.head      # index of syntactic head word
word.deprel    # dependency relation (nsubj, obj, advmod...)
```

## Multilingual
Same interface, different language code:
```python
stanza.download('de')
nlp = stanza.Pipeline('de', processors='tokenize,sentiment')
```
Models trained on Universal Dependencies treebanks — consistent annotation across languages.

## Key things to remember
- Sentiment is **per sentence**, not per document — average or aggregate yourself
- The model was trained on **movie reviews** — may underperform on other domains
- `sentiment` attribute only exists if the `sentiment` processor is loaded
- Stanza is slower than spaCy but more accurate, especially for morphologically complex languages
- multilingual ✓
- neural (CNN / Tree-LSTM) not rule-based like VADER ✓
- more accurate than VADER, especially on complex phrasing like negation ✓
- outputs per sentence, you iterate doc.sentences ✓



## Should I use spaCy for preprocessing and Stanza for inference?

The spaCy preprocessing + Stanza inference idea is interesting but you'd probably be creating more friction than you're solving — Stanza's preprocessors are already optimized to feed its own sentiment model. The tokenization and parse tree structure it builds internally is what the Tree-LSTM actually consumes. Swapping in spaCy's tokenizer could misalign the input format.

## Why are Neural Network based ones better than lexicon based ones?

### Lexicon (VADER):
- Looks up words one by one in a dictionary
- "not bad" → sees "bad" (-2.0) and "not" (modifier) → might still get it wrong
- No real understanding of how words interact
- Can't handle sarcasm at all — "oh great, another Monday" scores positive

### Neural Network (Stanza, BERT):
- Reads the whole sentence together
- Learned patterns from millions of examples, not hand-crafted rules
- Understands that "not bad" is positive because it saw that pattern in training data
- Can pick up on subtle context like tone and sarcasm (if trained on it)

### The tradeoff:
- Lexicon = fast, transparent, no training needed
- Neural = slower, needs training data, but much more accurate on complex language

Basically lexicons follow rules humans wrote, neural networks figure out their own rules 
from examples — and real language is too messy for hand-written rules to cover everything.