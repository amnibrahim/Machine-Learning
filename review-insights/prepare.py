import pandas as pd
from sklearn.model_selection import train_test_split

# for benchmarking the proposed model (transfer learning with distilbert)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline

SEED = 42

DATA_PATH = "data/JBL_T110BT_reviews.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    title = df["ReviewTitle"].fillna("").str.strip()
    body = df["ReviewBody"].fillna("").str.strip()
    df["text"] = (title + ". " + body).str.strip()
    return df[["text", "ReviewStar"]].rename(columns={"ReviewStar": "rating"})

def clean_and_label(df):
    df = df[df["rating"] != 3].copy()                 # drop neutral 3-star
    df["label"] = (df["rating"] >= 4).astype(int)     # 0 = negative, 1 = positive
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() >= 3]                # drop empty/near-empty
    df = df.drop_duplicates(subset="text")            # drop repeated reviews
    return df.reset_index(drop=True)

def split_data(df):
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=SEED
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )

# train (1127, 3) {1: 575, 0: 552}
# val (241, 3) {1: 123, 0: 118}
# test (242, 3) {1: 123, 0: 119}

def train_baseline(train, val):
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )
    model.fit(train["text"], train["label"])
    pred = model.predict(val["text"])
    metrics = {
        "accuracy": accuracy_score(val["label"], pred),
        "f1": f1_score(val["label"], pred),
        "confusion_matrix": confusion_matrix(val["label"], pred).tolist(),
    }
    return model, metrics

# {'accuracy': 0.9253112033195021, 'f1': 0.9256198347107438, 'confusion_matrix': [[111, 7], [11, 112]]}

# fine tuning:

import numpy as np
import torch
from transformers import set_seed
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256


def finetune_distilbert(train, val, epochs=3):
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    def to_dataset(df):
        ds = Dataset.from_pandas(df[["text", "label"]])
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
            batched=True,
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, pred),
            "f1": f1_score(labels, pred),
        }

    args = TrainingArguments(
        output_dir="models/distilbert",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=to_dataset(train),
        eval_dataset=to_dataset(val),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    return model, tokenizer, metrics

# evaluation:

import json
from pathlib import Path

ARTIFACTS = Path("artifacts")


def predict_distilbert(model, tokenizer, texts, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    probs = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(probs).numpy()          # probability of positive


def score(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_on_test(baseline, model, tokenizer, test):
    texts = test["text"].tolist()
    base_pred = baseline.predict(texts)
    bert_prob = predict_distilbert(model, tokenizer, texts)
    bert_pred = (bert_prob >= 0.5).astype(int)
    metrics = {
        "baseline": score(test["label"], base_pred),
        "distilbert": score(test["label"], bert_pred),
        "n_test": len(test),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    out = test.copy()
    out["pred"] = bert_pred
    out["prob_pos"] = bert_prob.round(4)
    return metrics, out

#                                                 text  rating  label  pred  prob_pos
# 0  Worst Sound Quality. Returned after using for ...       1      0     0    0.0166
# 1  EARPHONES NOT CHARGING. My earphones aren’t ch...       1      0     0    0.0213
# 2         Perfect but for money!. Worth the money!!!       5      1     1    0.9807
# 3     Four Stars. Good bt Base is so so.... Not good       4      1     0    0.4821
# 4  Its totally tangle free. It's great I have bee...       5      1     1    0.9772

# SHAP implementation for explaniablity

import time
import shap
from transformers import pipeline

def explain_texts(model, tokenizer, texts, preds):
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,                      # return scores for both classes
        device=next(model.parameters()).device,
        truncation=True,
        max_length=MAX_LEN,
    )
    explainer = shap.Explainer(clf)
    sv = explainer(texts)
    results = []
    for i, pred in enumerate(preds):
        tokens = [t.strip() for t in sv.data[i]]
        values = sv.values[i][:, pred]   # scores toward the predicted class
        results.append(list(zip(tokens, values.round(4).tolist())))
    return results

# save to local

from collections import Counter

def load_or_train(train, val):
    path = Path("models/distilbert_final")
    if path.exists():
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return model.to("cuda" if torch.cuda.is_available() else "cpu"), tok
    model, tok, _ = finetune_distilbert(train, val)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return model, tok

def explain_all(model, tok, test_out, chunk=20):
    cache = ARTIFACTS / "shap_cache.json"
    done = json.loads(cache.read_text()) if cache.exists() else []
    texts = test_out["text"].tolist()
    preds = test_out["pred"].tolist()
    for i in range(len(done), len(texts), chunk):
        done += explain_texts(model, tok, texts[i : i + chunk], preds[i : i + chunk])
        cache.write_text(json.dumps(done))
        print(f"SHAP {len(done)}/{len(texts)}")
    return done

if __name__ == "__main__":
    train, val, test = split_data(clean_and_label(load_data()))
    baseline, _ = train_baseline(train, val)
    model, tok = load_or_train(train, val)
    metrics, test_out = evaluate_on_test(baseline, model, tok, test)

    shap_tokens = explain_all(model, tok, test_out)
    test_out["shap"] = [json.dumps(t) for t in shap_tokens]
    test_out.to_csv(ARTIFACTS / "test_pool.csv", index=False)

    for cls, name in [(0, "negative"), (1, "positive")]:
        totals = Counter()
        for s in test_out.loc[test_out["pred"] == cls, "shap"]:
            for tok_, val_ in json.loads(s):
                totals[tok_.lower()] += val_
        print(f"\n{name}:", [(w, round(v, 2)) for w, v in totals.most_common(25)])