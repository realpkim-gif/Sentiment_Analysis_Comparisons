import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 64


def predict(df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    def tokenize_batch(batch):
        return tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # the DataLoader hands us the tweets in small batches (in order, so predictions line up with df)
    loader = DataLoader(df["clean_text"].tolist(), batch_size=BATCH_SIZE, shuffle=False, collate_fn=tokenize_batch)
    predictions = []

    for inputs in loader:
        inputs = inputs.to(device)

        with torch.no_grad():  # no training, so skip gradients (faster, less memory)
            logits = model(**inputs).logits

        # this model's labels are already "negative" / "neutral" / "positive", so no renaming needed
        for class_id in logits.argmax(dim=-1).tolist():
            predictions.append(model.config.id2label[class_id])

    return pd.Series(predictions, index=df.index)
