import json
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import pandas as pd

MIN_SCORE = 0.0       # keep any token that pushed toward the predicted class
TOP_N = 25            # keywords kept per class (20-30)
N_SNIPPETS = 3        # snippets per class (equal for both classes)
SNIPPET_CHARS = 200

# since not and no are some of top negative signals, keep these:
STOP = set(ENGLISH_STOP_WORDS) - {"not", "no", "never", "nor", "cannot"}

def sample_batch(pool, n_per_class=10, seed=None):
    """Draw n reviews per true label from the test pool."""
    parts = [g.sample(n_per_class, random_state=seed) for _, g in pool.groupby("label")]
    return pd.concat(parts).reset_index(drop=True)


def aggregate_keywords(batch):
    """Per predicted class: top keywords (review count, summed SHAP) and equal-sized snippets."""
    classes = {0: "negative", 1: "positive"}
    n_snip = min(N_SNIPPETS, *(int((batch["pred"] == c).sum()) for c in classes))
    out = {}
    for c, name in classes.items():
        rows = batch[batch["pred"] == c]
        count = defaultdict(int)
        total = defaultdict(float)
        for s in rows["shap"]:
            seen = set()
            for tok, val in json.loads(s):
                w = tok.lower()
                if val < MIN_SCORE or not w.isalpha() or len(w) < 2 or w in STOP:
                    continue
                total[w] += val
                seen.add(w)
            for w in seen:
                count[w] += 1
        ranked = sorted(total, key=total.get, reverse=True)[:TOP_N]
        # keywords = [(w, count[w], round(total[w], 2)) for w in ranked]
        keywords = [
            {"word": w, "reviews_containing": count[w], "total_shap": round(total[w], 2)}
            for w in ranked
        ]
        conf = rows["prob_pos"] if c == 1 else 1 - rows["prob_pos"]
        top = rows.assign(conf=conf).sort_values("conf", ascending=False).head(n_snip)
        snippets = [t[:SNIPPET_CHARS] for t in top["text"]]
        # out[name] = {"n_reviews": len(rows), "keywords": keywords, "snippets": snippets}
        out[name] = {"n_reviews": len(rows), "keywords": keywords, "example_reviews": snippets}
        out["sample_size"] = len(batch)
    return out


if __name__ == "__main__":
    pool = pd.read_csv("artifacts/test_pool.csv")
    for seed in (1, 2):
        result = aggregate_keywords(sample_batch(pool, seed=seed))
        print(f"\n=== seed {seed} ===")
        print(json.dumps(result, indent=2))