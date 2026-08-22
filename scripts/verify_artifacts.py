"""Verify committed model artifacts and run a deterministic smoke inference.

Usage: python scripts/verify_artifacts.py
"""
from pathlib import Path
import json
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "category_classifier.pkl",
    "tfidf_vectorizer.pkl",
    "suitability_model.pkl",
    "suitability_vectorizer.pkl",
    "skill_vectorizer.pkl",
    "model_manifest.json",
]


def main() -> int:
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    if missing:
        print(f"Missing artifacts: {', '.join(missing)}", file=sys.stderr)
        return 1

    manifest = json.loads((ROOT / "model_manifest.json").read_text())
    classifier = joblib.load(ROOT / "category_classifier.pkl")
    vectorizer = joblib.load(ROOT / "tfidf_vectorizer.pkl")
    suitability = joblib.load(ROOT / "suitability_model.pkl")
    suitability_vectorizer = joblib.load(ROOT / "suitability_vectorizer.pkl")
    skill_vectorizer = joblib.load(ROOT / "skill_vectorizer.pkl")

    sample = "Python FastAPI machine learning data analysis"
    matrix = vectorizer.transform([sample])
    category = classifier.predict(matrix)[0]
    suitability_matrix = suitability_vectorizer.transform([sample])
    skill_matrix = skill_vectorizer.transform([sample])

    checks = {
        "classifier_classes": len(getattr(classifier, "classes_", [])),
        "category_prediction": str(category),
        "category_vector_width": matrix.shape[1],
        "suitability_vector_width": suitability_matrix.shape[1],
        "skill_vector_width": skill_matrix.shape[1],
        "suitability_features": getattr(suitability, "n_features_in_", None),
        "pipeline_version": manifest["pipeline_version"],
    }
    if checks["classifier_classes"] != manifest["stage_1"]["classes"]:
        print("Classifier class count does not match model_manifest.json", file=sys.stderr)
        return 1
    print(json.dumps(checks, indent=2))
    print("Artifact verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
