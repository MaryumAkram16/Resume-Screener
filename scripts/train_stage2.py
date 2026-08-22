"""Reproduce the documented Stage 2 suitability model.

The source must be a CSV exported from batuhanmtl/job_resume_fit with the
columns used in resume_screener.ipynb. The script deliberately records that
synthetic negative pairs are heuristic labels, not human ground truth.

Usage:
  python scripts/train_stage2.py --csv job_resume_fit.csv --output-dir artifacts/stage2
"""
from __future__ import annotations
import argparse, ast, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
FEATURES = ["skill_overlap_ratio", "skill_overlap_count", "length_ratio", "text_similarity", "skill_text_similarity", "embedding_similarity", "skill_string_match_score", "fuzzy_match_score"]

def parse_list(value):
    try:
        result = ast.literal_eval(value) if isinstance(value, str) else value
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []

def negative_pairs(df, n):
    rng = np.random.RandomState(SEED); rows = []
    for _ in range(n):
        resume = df.iloc[rng.randint(len(df))]; candidates = df[df["category"] != resume["category"]]
        if candidates.empty: continue
        job = candidates.iloc[rng.randint(len(candidates))]
        rows.append({"ID": f"neg_{resume['ID']}_{job['ID']}", "resume_text": resume["resume_text"], "job_text": job["job_text"], "category": job["category"], "resume_skills_parsed": resume["resume_skills_parsed"], "job_skills_parsed": job["job_skills_parsed"], "ai_match_score": rng.uniform(0, 15), "skill_string_match_score": rng.uniform(0, 5), "fuzzy_match_score": rng.uniform(10, 30)})
    return pd.DataFrame(rows)

def add_features(df, embedding_model=None):
    df = df.copy(); df["resume_text"] = df["resume_text"].fillna("").astype(str); df["job_text"] = df["job_text"].fillna("").astype(str)
    df["resume_skills_parsed"] = df["resume_skills_parsed"].map(parse_list); df["job_skills_parsed"] = df["job_skills_parsed"].map(parse_list)
    df["skill_overlap_ratio"] = df.apply(lambda r: len(set(map(str.lower, r["resume_skills_parsed"])) & set(map(str.lower, r["job_skills_parsed"]))) / max(len(set(map(str.lower, r["job_skills_parsed"]))), 1), axis=1)
    df["skill_overlap_count"] = df.apply(lambda r: len(set(map(str.lower, r["resume_skills_parsed"])) & set(map(str.lower, r["job_skills_parsed"]))), axis=1)
    df["length_ratio"] = df.apply(lambda r: min(len(r["resume_text"].split()), len(r["job_text"].split())) / max(len(r["resume_text"].split()), len(r["job_text"].split()), 1), axis=1)
    text_vec = TfidfVectorizer(stop_words="english", max_features=8000); matrix = text_vec.fit_transform(pd.concat([df["resume_text"], df["job_text"]])); n = len(df)
    df["text_similarity"] = [cosine_similarity(matrix[i], matrix[n+i])[0][0] for i in range(n)]
    df["resume_skills_text"] = df["resume_skills_parsed"].map(lambda x: " ".join(map(str, x))); df["job_skills_text"] = df["job_skills_parsed"].map(lambda x: " ".join(map(str, x)))
    skill_vec = TfidfVectorizer(stop_words="english", max_features=4000); skill_matrix = skill_vec.fit_transform(pd.concat([df["resume_skills_text"], df["job_skills_text"]]));
    df["skill_text_similarity"] = [cosine_similarity(skill_matrix[i], skill_matrix[n+i])[0][0] for i in range(n)]
    if embedding_model is not None:
        resume_embeddings = embedding_model.encode(df["resume_text"].tolist(), show_progress_bar=False)
        job_embeddings = embedding_model.encode(df["job_text"].tolist(), show_progress_bar=False)
        df["embedding_similarity"] = [cosine_similarity([resume_embeddings[i]], [job_embeddings[i]])[0][0] for i in range(n)]
    else:
        # Explicit lightweight fallback for environments without the optional model.
        df["embedding_similarity"] = df["text_similarity"]
    df["_skill_vocab"] = ["" for _ in range(n)]
    return df, text_vec, skill_vec

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv", type=Path, required=True); p.add_argument("--output-dir", type=Path, default=Path("artifacts/stage2")); p.add_argument("--no-embeddings", action="store_true", help="use text similarity as a lightweight fallback"); args = p.parse_args()
    raw = pd.read_csv(args.csv); required = {"ID", "resume_text", "job_text", "category", "resume_skill_list", "job_required_skills", "ai_match_score"}; missing = required - set(raw.columns)
    if missing: raise SystemExit(f"Missing required columns: {sorted(missing)}")
    raw["resume_skills_parsed"] = raw["resume_skill_list"].map(parse_list); raw["job_skills_parsed"] = raw["job_required_skills"].map(parse_list)
    neg = negative_pairs(raw, len(raw)//3); full = pd.concat([raw, neg], ignore_index=True, sort=False); full["skill_string_match_score"] = full["skill_string_match_score"].fillna(0); full["fuzzy_match_score"] = full["fuzzy_match_score"].fillna(0)
    embedding_model = None
    if not args.no_embeddings:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    full, text_vec, skill_vec = add_features(full, embedding_model)
    x_train, x_test, y_train, y_test = train_test_split(full[FEATURES], full["ai_match_score"], test_size=0.2, random_state=SEED)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED); model.fit(x_train, y_train); pred = model.predict(x_test)
    args.output_dir.mkdir(parents=True, exist_ok=True); joblib.dump(model, args.output_dir / "suitability_model.pkl"); joblib.dump(text_vec, args.output_dir / "suitability_vectorizer.pkl"); joblib.dump(skill_vec, args.output_dir / "skill_vectorizer.pkl")
    metrics = {"seed": SEED, "source_rows": int(len(raw)), "synthetic_negative_rows": int(len(neg)), "combined_rows": int(len(full)), "mae": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred)), "features": FEATURES, "embedding_feature_note": "all-MiniLM-L6-v2 when default mode is used; text-similarity fallback only when --no-embeddings is passed", "labels_note": "negative-pair scores are heuristic synthetic labels"}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2)); print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
