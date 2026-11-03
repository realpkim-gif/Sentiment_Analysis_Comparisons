import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "google-bert/bert-base-uncased"
BATCH_SIZE = 64

# plain BERT was never trained on sentiment, it only fills in blanks ([MASK]).
# so we add a sentence after each tweet and see which word BERT thinks fits the blank best.
PROMPT = "This is [MASK]."

# the three words BERT can choose from for the blank
# (this prompt + these words scored best when we tried a few on sample tweets)
WORDS = ["terrible", "normal", "great"]


def predict(df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    word_ids = tokenizer.convert_tokens_to_ids(WORDS)

    def tokenize_batch(batch):
        # "only_first" trims long tweets but never cuts off the prompt (and its [MASK])
        return tokenizer(batch, [PROMPT] * len(batch), padding=True, truncation="only_first",
                         max_length=128, return_tensors="pt")

    # the DataLoader hands us the tweets in small batches (in order, so predictions line up with df)
    loader = DataLoader(df["clean_text"].tolist(), batch_size=BATCH_SIZE, shuffle=False, collate_fn=tokenize_batch)
    predictions = []

    for inputs in loader:
        inputs = inputs.to(device)

        with torch.no_grad():  # no training, so skip gradients (faster, less memory)
            logits = model(**inputs).logits

        # BERT's scores for the blank in each tweet, for just our three words
        blank_scores = logits[inputs["input_ids"] == tokenizer.mask_token_id][:, word_ids]  # only keeps spots where True (the [MASK]), then only our 3 words' scores

        # for each tweet, the position of the highest-scoring word in WORDS (0, 1 or 2)
        best_words = blank_scores.argmax(dim=-1).tolist()

        for best_word in best_words:
            if best_word == 0:  # "terrible"
                predictions.append("negative")
            elif best_word == 1:  # "normal"
                predictions.append("neutral")
            else:  # "great"
                predictions.append("positive")

    return pd.Series(predictions, index=df.index)
