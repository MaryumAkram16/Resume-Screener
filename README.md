# Resume Screener

A two-stage system that predicts a resume's job category, then scores how well it actually fits a specific job description — trained on a small, real-world resume dataset and deployed as an interactive Streamlit app.

## 🚀 Live App

[resume-screener-prediction.streamlit.app](https://resume-screener-prediction.streamlit.app/)

## Problem

Screening resumes against open roles is normally a manual, time-consuming process. This project builds a lightweight two-stage pipeline that automates the first pass:

1. **Stage 1 — Category Classifier**: predicts what kind of role a resume belongs to (HR, IT, Finance, Chef, etc.)
2. **Stage 2 — Suitability Scoring**: given a resume and a specific job description, scores how good a fit it actually is — not just which category it belongs to, but how well its content matches the posting

## Dataset

### Stage 1 — Resumes

- **Source**: [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (Kaggle)
- **Size**: 2,484 resumes across 24 job categories
- **Columns used**: `Resume_str` (raw resume text), `Category` (label)

The raw CSV isn't included in this repo (GitHub has file size limits, and it's easy to grab directly from Kaggle). To reproduce training: download `Resume.csv` from the link above and upload it when the notebook prompts for it.

### Stage 2 — Resume-Job Pairs

- **Source**: [job_resume_fit](https://huggingface.co/datasets/batuhanmtl/job_resume_fit) (Hugging Face)
- **Size**: 2,385 resume-job posting pairs across 23 categories, built from the same source resume dataset as Stage 1
- **Columns used**: `resume_text`, `job_text`, `category`, `job_required_skills`, `resume_skill_list`, `ai_match_score` (training target), `skill_string_match_score`, `fuzzy_match_score`

This dataset is pulled directly in the notebook via `datasets.load_dataset("batuhanmtl/job_resume_fit")` — no manual download needed.

### Data Cleaning (Stage 1)

- Dropped 2 pairs of exact duplicate resumes and 1 near-empty resume (under 10 words)
- Stripped stray HTML tags and collapsed irregular whitespace left over from the original scrape
- Merged `Consultant`, `Business-Development`, and `Sales` into one `Business` category — the confusion matrix showed these were constantly mistaken for each other, since the language genuinely overlaps
- Dropped `Automobile` and `BPO` — too few samples (under 40) to learn a reliable pattern from
- Result: 24 raw categories → 20 usable categories

## Stage 1 — Category Classifier

- **Vectorizer**: `TfidfVectorizer` (unigrams + bigrams, English stop words removed, max 10,000 features)
- **Classifier**: `LogisticRegression` (`class_weight='balanced'`, `solver='lbfgs'`)

### Results

**Accuracy: 68.0%**

| Category | Precision | Recall | F1-score |
|---|---|---|---|
| Information-Technology | 0.64 | 0.96 | 0.77 |
| HR | 0.83 | 0.91 | 0.87 |
| Fitness | 0.94 | 0.71 | 0.81 |
| Construction | 0.85 | 0.77 | 0.81 |
| Business | 0.66 | 0.50 | 0.57 |
| Apparel | 0.44 | 0.42 | 0.43 |

Full per-category breakdown: [`table_classification_report.csv`](table_classification_report.csv)

A random baseline across 20 categories would score ~5% — 68% reflects real learned signal, with the main bottleneck being dataset size (~2,400 resumes) rather than model choice.

### Confusion Matrix

![Confusion Matrix](chart_confusion_matrix.png)

Most remaining confusion is concentrated in categories with genuine content overlap — see [`table_confusion_mistakes.csv`](table_confusion_mistakes.csv) for the full breakdown.

## Stage 2 — Suitability Scoring

The `job_resume_fit` dataset only pairs resumes with jobs they were actually matched against — every real example is some flavor of "decent fit." To teach the model what a *bad* fit looks like, synthetic negative pairs were added: resumes paired with a job from a different category, labeled with a low synthetic score (0-15). This is a heuristic assumption, not ground-truth data, but it meaningfully improved the model — see results below.

### Features (8 total)

| Feature | Description |
|---|---|
| `skill_overlap_ratio` | % of the job's required skills found in the resume's skill list |
| `skill_overlap_count` | Raw count of overlapping skills |
| `length_ratio` | How close resume and job description lengths are |
| `text_similarity` | TF-IDF cosine similarity between full resume and job text |
| `skill_text_similarity` | TF-IDF cosine similarity between just the skill lists |
| `embedding_similarity` | Sentence-transformer (`all-MiniLM-L6-v2`) cosine similarity between resume and job text |
| `skill_string_match_score` | Provided by the dataset (original author's scoring pipeline) |
| `fuzzy_match_score` | Provided by the dataset (original author's scoring pipeline) |

### Model Comparison

| Model | MAE | R² |
|---|---|---|
| **Gradient Boosting (final)** | **8.69** | **0.814** |
| Tuned Random Forest (GridSearchCV) | 8.98 | 0.801 |
| Random Forest (default) | 9.04 | 0.799 |

Full comparison: [`table_model_comparison.csv`](table_model_comparison.csv)

### Predicted vs Actual

![Predicted vs Actual](chart_predicted_vs_actual.png)

### Feature Importance

![Feature Importance](chart_feature_importance.png)

Plain TF-IDF `text_similarity` ended up the dominant feature (~47% importance) — the sentence-embedding feature, expected to add meaningful semantic understanding, contributed only ~6%. This suggests resume-job fit in this dataset is driven more by shared vocabulary than by paraphrased meaning — a useful negative result, not a bug.

### What the negative pairs fixed

Before adding synthetic negative examples, HR and Information-Technology had the worst per-category error (20.7 and 18.5 average absolute error) despite being among the *best*-classified categories in Stage 1 — likely because these categories have the widest real-world range of "good" vs "bad" fit, and the model had nothing to contrast against. After adding negative pairs, HR's error dropped to ~11.5. Full breakdown: [`table_category_error.csv`](table_category_error.csv).

### A real test case

| Test | Suitability Score |
|---|---|
| HR resume vs. genuine HR job posting | 51.8 — Moderate fit |
| Same HR resume vs. unrelated Chef job posting | 2.8 — Weak fit |

A ~49-point gap between a genuine match and a deliberate mismatch, using the deployed app itself — concrete evidence the model is picking up on real content alignment, not just rewarding generic professional-sounding text.

## Honest Limitations

- **Approximated match scores**: `skill_string_match_score` and `fuzzy_match_score` in the training data came from the original dataset author's private scoring pipeline. The live app's `score_resume_against_job()` function uses its own honest approximations (word-boundary skill matching, `difflib` fuzzy matching) — not the exact original formula.
- **Small training set**: 2,484 resumes across 20+ categories means some categories (Agriculture, Apparel) have only a few dozen examples. 68% Stage 1 accuracy reflects a genuinely hard, data-limited problem more than a modeling shortfall.
- **Category errors can propagate**: if Stage 1 misclassifies a resume's category, that mistake doesn't feed directly into Stage 2's features, but a systematically wrong category signal upstream can still color how results should be interpreted downstream.

## App Features

- **Try It**: paste a resume and job description, get a predicted category (with confidence breakdown) and a suitability score
- **Model Performance**: live charts and metrics for both stages
- **Model & Method**: full pipeline walkthrough, key engineering decisions, and the limitations above

## Usage

```python
import joblib
from sentence_transformers import SentenceTransformer

clf = joblib.load('category_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
suitability_model = joblib.load('suitability_model.pkl')
vectorizer_s2 = joblib.load('suitability_vectorizer.pkl')
skill_vectorizer = joblib.load('skill_vectorizer.pkl')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# see app.py for the full score_resume_against_job() implementation
```

## Repository

[github.com/MaryumAkram16/Resume-Screener](https://github.com/MaryumAkram16/Resume-Screener)

## Tech Stack

- `scikit-learn` — TF-IDF, Logistic Regression, Random Forest, Gradient Boosting
- `sentence-transformers` — semantic embedding similarity (`all-MiniLM-L6-v2`)
- `datasets` (Hugging Face) — loading the resume-job pairs dataset
- `pandas` / `numpy` — data handling
- `matplotlib` / `seaborn` — visualization
- `streamlit` — deployed app interface
- `joblib` — model persistence

## Possible Next Steps

- Recreate `skill_string_match_score` and `fuzzy_match_score` from scratch with a documented formula, rather than relying on the original dataset's black-box versions
- Expand the resume dataset (more categories, more samples per category) to push Stage 1 accuracy past the current data-limited ceiling
- Feed Stage 1's predicted category into Stage 2 as an explicit feature for a brand-new (unlabeled) resume, rather than only using it at inference time for display


## Reproducibility and model lifecycle

The committed `.pkl` files are the versioned inference artifacts used by the deployed app. Their pipeline metadata is recorded in [`model_manifest.json`](model_manifest.json), including the training seed, dataset sources, feature contract, artifact names, and reported metrics.

Verify the artifacts and run a deterministic smoke inference locally:

```bash
pip install -r requirements.txt
python scripts/verify_artifacts.py
```

The same check runs on every push and pull request through [`.github/workflows/verify.yml`](.github/workflows/verify.yml). The Docker image preserves the current Streamlit entrypoint:

```bash
docker build -t resume-screener:local .
docker run --rm -p 8501:8501 resume-screener:local
```

Retraining still requires the external datasets described above. The manifest separates the committed model version from future retraining runs so new artifacts can be compared without silently changing the deployed inference contract.
