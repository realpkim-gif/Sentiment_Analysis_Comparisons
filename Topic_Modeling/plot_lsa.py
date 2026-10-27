import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
results = pd.read_csv(os.path.join(HERE, "lsa_explained_variance.csv"))

plt.plot(results["n_components"], results["explained_variance"], marker="o")
plt.xlabel("n_components")
plt.ylabel("Explained variance")
plt.title("LSA explained variance")
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
plt.savefig(os.path.join(HERE, "plots", "lsa_explained_variance.png"), dpi=150)
