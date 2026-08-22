"""Reproduce the Stage 1 classifier from a downloaded Resume.csv.

Usage:
  python scripts/train_stage1.py --csv /path/to/Resume.csv --output-dir artifacts/stage1
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

SEED = 42
DROP = {"Automobile", "BPO"}
MERGE = {"Consultant": "Business", "Business-Development": "Business", "Sales": "Business"}

def clean(text: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip()

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--csv", type=Path, required=True); p.add_argument("--output-dir", type=Path, default=Path("artifacts/stage1")); args = p.parse_args()
    df = pd.read_csv(args.csv)
    df["Resume_str"] = df["Resume_str"].map(clean)
    df = df[df["Resume_str"].str.split().str.len() >= 10].drop_duplicates("Resume_str").copy()
    df["Category"] = df["Category"].replace(MERGE)
    df = df[~df["Category"].isin(DROP)].copy()
    x_train, x_test, y_train, y_test = train_test_split(df["Resume_str"], df["Category"], test_size=0.2, random_state=SEED, stratify=df["Category"])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=10000)
    x_train_vec = vectorizer.fit_transform(x_train); x_test_vec = vectorizer.transform(x_test)
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=SEED)
    clf.fit(x_train_vec, y_train); pred = clf.predict(x_test_vec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.output_dir / "category_classifier.pkl"); joblib.dump(vectorizer, args.output_dir / "tfidf_vectorizer.pkl")
    metrics = {"seed": SEED, "rows": int(len(df)), "classes": int(df["Category"].nunique()), "accuracy": float(accuracy_score(y_test, pred)), "report": classification_report(y_test, pred, output_dict=True, zero_division=0)}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("seed", "rows", "classes", "accuracy")}, indent=2))

if __name__ == "__main__": main()
