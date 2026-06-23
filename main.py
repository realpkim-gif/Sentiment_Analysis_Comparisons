import pandas as pd
from models import vader_model, textblob_model, spacy_model, afinn_model, sentiwordnet_model, flair_model, stanza_sentiment

df = pd.read_csv(r"dataFolder/train.csv", encoding="latin-1")
df = df[["text", "sentiment"]].fillna("")

results = []
results.append(vader_model.get_accuracy(df.copy()))
results.append(textblob_model.get_accuracy(df.copy()))
results.append(afinn_model.get_accuracy(df.copy()))
results.append(spacy_model.get_accuracy(df.copy()))
results.append(sentiwordnet_model.get_accuracy(df.copy()))
results.append(flair_model.get_accuracy(df.copy()))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
print(results_df)
