"""Exploratory analysis of the Yelp Open Dataset (https://business.yelp.com/data/resources/open-dataset/).

Download the JSON version, extract it, and put the files in Topic_Modeling/yelp_data/ :
    yelp_academic_dataset_business.json
    yelp_academic_dataset_review.json      (~5 GB, read in chunks)
    yelp_academic_dataset_user.json        (optional)
    yelp_academic_dataset_tip.json         (optional)

Run:  python Topic_Modeling/yelp_eda.py
Charts are saved to Topic_Modeling/plots/.
"""
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "yelp_data")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

BUSINESS = os.path.join(DATA, "yelp_academic_dataset_business.json")
REVIEW = os.path.join(DATA, "yelp_academic_dataset_review.json")


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, name), dpi=150)
    plt.close()


# ------------------------------------------------------------------ business
biz = pd.read_json(BUSINESS, lines=True)
print("\n=== business.json ===")
print("shape:", biz.shape)
print("columns:", list(biz.columns))
print(biz.dtypes)
print("missing values:\n", biz.isna().sum())
print("unique businesses:", biz["business_id"].nunique())

# categories: comma separated string -> list
biz["cat_list"] = biz["categories"].fillna("").str.split(", ")
cat_counts = Counter(c for cats in biz["cat_list"] for c in cats if c)
print("distinct categories:", len(cat_counts))
biz["is_restaurant"] = biz["cat_list"].apply(lambda c: "Restaurants" in c)
print("restaurants:", biz["is_restaurant"].sum(), "of", len(biz))
print("open vs closed:\n", biz["is_open"].value_counts())

top_cats = pd.Series(cat_counts).sort_values(ascending=False).head(25)
top_cats.sort_values().plot.barh(figsize=(8, 7), title="Top 25 business categories")
save("01_top_categories.png")

biz["state"].value_counts().head(15).sort_values().plot.barh(figsize=(7, 5), title="Businesses by state (top 15)")
save("02_businesses_by_state.png")

biz["city"].value_counts().head(15).sort_values().plot.barh(figsize=(7, 5), title="Businesses by city (top 15)")
save("03_businesses_by_city.png")

biz["stars"].value_counts().sort_index().plot.bar(figsize=(7, 4), title="Business average star rating")
save("04_business_stars.png")

biz["review_count"].clip(upper=500).plot.hist(bins=50, figsize=(7, 4), title="Reviews per business (clipped at 500)")
save("05_reviews_per_business.png")

# restaurant sub-categories (everything co-listed with "Restaurants")
rest = biz[biz["is_restaurant"]]
rest_cats = Counter(c for cats in rest["cat_list"] for c in cats if c and c not in ("Restaurants", "Food"))
pd.Series(rest_cats).sort_values(ascending=False).head(20).sort_values().plot.barh(
    figsize=(8, 6), title="Top 20 restaurant sub-categories")
save("06_restaurant_subcategories.png")

print("\nTop restaurant states:\n", rest["state"].value_counts().head(10))

# ------------------------------------------------------------------- reviews
print("\n=== review.json (chunked) ===")
rest_ids = set(rest["business_id"])
by_year = Counter()
stars_all, stars_rest = Counter(), Counter()
length_sample = []
n_all = n_rest = 0
min_date, max_date = None, None

for chunk in pd.read_json(REVIEW, lines=True, chunksize=200_000):
    n_all += len(chunk)
    d = pd.to_datetime(chunk["date"])
    min_date = d.min() if min_date is None else min(min_date, d.min())
    max_date = d.max() if max_date is None else max(max_date, d.max())
    stars_all.update(chunk["stars"])
    is_rest = chunk["business_id"].isin(rest_ids)
    n_rest += is_rest.sum()
    stars_rest.update(chunk.loc[is_rest, "stars"])
    by_year.update(d[is_rest].dt.year)
    length_sample.extend(chunk.loc[is_rest, "text"].str.split().str.len().sample(frac=0.02, random_state=0))

print("columns:", list(chunk.columns))
print(f"total reviews: {n_all:,} | restaurant reviews: {n_rest:,}")
print("date range:", min_date.date(), "->", max_date.date())

pd.Series(by_year).sort_index().plot.bar(figsize=(9, 4), title="Restaurant reviews per year")
save("07_reviews_per_year.png")

pd.DataFrame({"all": pd.Series(stars_all), "restaurants": pd.Series(stars_rest)}).sort_index().plot.bar(
    figsize=(7, 4), title="Review star distribution")
save("08_review_stars.png")

pd.Series(length_sample).clip(upper=800).plot.hist(bins=60, figsize=(7, 4),
                                                   title="Review length in words (restaurants, 2% sample, clipped at 800)")
save("09_review_length.png")
print("median review length (words):", pd.Series(length_sample).median())

# optional files
for name in ("user", "tip"):
    path = os.path.join(DATA, f"yelp_academic_dataset_{name}.json")
    if os.path.exists(path):
        df = next(pd.read_json(path, lines=True, chunksize=100_000))
        print(f"\n=== {name}.json (first chunk) ===\ncolumns: {list(df.columns)}")

print("\nCharts saved to", PLOTS)
