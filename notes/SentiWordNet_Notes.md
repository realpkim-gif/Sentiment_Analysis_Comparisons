# SentiWordNet — Quick Notes

## What it is
SentiWordNet is WordNet's synsets, each tagged with three scores: `pos_score`, `neg_score`, `obj_score` (sum to 1). It's a raw lexical resource, not a finished sentiment analyzer.

## How the pipeline used it
1. POS tag each word (Penn tags: `NN`, `VBD`, `JJ`...)
2. Convert Penn tags → WordNet tags (`n`, `v`, `a`, `r`) via `penn_to_wn`
3. Lemmatize using that POS, so the word reduces to its correct base form
4. Look up the lemma's first/most common synset in WordNet
5. Pull that synset's `pos_score` / `neg_score` from SentiWordNet
6. Sum `pos - neg` across all words in the sentence → overall score

## Why it's trash
- No negation handling ("not good" scores as positive)
- No intensifiers ("very bad" = "bad")
- First-sense lookup is often the wrong sense in context
- Sparse coverage of slang/informal text (tweets especially)
- Naive summation ignores sentence structure entirely

## How it compares
- **VADER** = SentiWordNet's idea (lexicon of sentiment-scored words) **+ rules**: negation, intensifiers, punctuation, capitalization, social-media-tuned vocabulary. Works on raw text directly.
- **TextBlob** = similar lexicon + rules approach, but lighter/less tuned than VADER. Also gives a subjectivity score.
- Both are essentially "SentiWordNet, but with the missing context-handling rules bolted on."

## Bottom line
Don't memorize the API. The transferable lessons are: lexicon-based, word-by-word scoring throws away context, and that's why it's weak. The pipeline shape (clean → tokenize → POS tag → lemmatize → score → aggregate) is the reusable part.

## Something to rememeber:
It's just a VADER with no rules. Looks up sentiments and you can do sentiment analysis by averaging the scores manually. Thats it! Dont overthink. The POS tag is only used to give context during lemmetization so it correctly lemmetizes the right way.
