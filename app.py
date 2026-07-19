import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Resume Screener",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURE_COLS = [
    "skill_overlap_ratio", "skill_overlap_count", "length_ratio",
    "text_similarity", "skill_text_similarity", "embedding_similarity",
    "skill_string_match_score", "fuzzy_match_score",
]


@st.cache_resource
def load_models():
    clf = joblib.load("category_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    suitability_model = joblib.load("suitability_model.pkl")
    vectorizer_s2 = joblib.load("suitability_vectorizer.pkl")
    skill_vectorizer = joblib.load("skill_vectorizer.pkl")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return clf, vectorizer, suitability_model, vectorizer_s2, skill_vectorizer, embed_model


clf, vectorizer, suitability_model, vectorizer_s2, skill_vectorizer, embed_model = load_models()
SKILL_VOCAB = skill_vectorizer.get_feature_names_out()


def extract_skills(text, vocab=SKILL_VOCAB):
    """Word-boundary matching against the skill vocabulary the skill_vectorizer
    was fit on during training - not a fancy NER model, just honest keyword
    matching against real skills seen in the training data."""
    text_lower = text.lower()
    return [skill for skill in vocab if re.search(r"\b" + re.escape(skill) + r"\b", text_lower)]


def skill_string_match_stub(resume_text, job_skills):
    """Stand-in for skill_string_match_score - the original dataset's version
    came from the author's own RecAI pipeline, which isn't public. This is a
    simple honest substitute: % of the job's required skills found verbatim
    (whole word/phrase) in the resume text."""
    if not job_skills:
        return 0.0
    text_lower = resume_text.lower()
    matches = sum(1 for s in job_skills if re.search(r"\b" + re.escape(s) + r"\b", text_lower))
    return 100 * matches / len(job_skills)


def fuzzy_match_stub(resume_skills, job_skills):
    """Stand-in for fuzzy_match_score - uses difflib's sequence matcher on the
    joined skill strings instead of the original's fuzzy-token method."""
    resume_str = " ".join(resume_skills)
    job_str = " ".join(job_skills)
    if not resume_str or not job_str:
        return 0.0
    return SequenceMatcher(None, resume_str, job_str).ratio() * 100


def score_resume_against_job(resume_text, job_text):
    proba = clf.predict_proba(vectorizer.transform([resume_text]))[0]
    classes = clf.classes_
    order = np.argsort(proba)[::-1]
    category = classes[order[0]]

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    resume_set, job_set = set(resume_skills), set(job_skills)

    skill_overlap_ratio_val = len(resume_set & job_set) / len(job_set) if job_set else 0.0
    skill_overlap_count_val = len(resume_set & job_set)

    resume_word_count = len(resume_text.split())
    job_word_count = len(job_text.split())
    length_ratio_val = min(resume_word_count, job_word_count) / max(resume_word_count, job_word_count, 1)

    text_sim = cosine_similarity(
        vectorizer_s2.transform([resume_text]), vectorizer_s2.transform([job_text])
    )[0][0]

    skill_text_sim = cosine_similarity(
        skill_vectorizer.transform([" ".join(resume_skills)]),
        skill_vectorizer.transform([" ".join(job_skills)])
    )[0][0]

    resume_emb = embed_model.encode([resume_text])
    job_emb = embed_model.encode([job_text])
    embedding_sim = cosine_similarity(resume_emb, job_emb)[0][0]

    skill_string_score = skill_string_match_stub(resume_text, job_skills)
    fuzzy_score = fuzzy_match_stub(resume_skills, job_skills)

    features_df = pd.DataFrame([[
        skill_overlap_ratio_val, skill_overlap_count_val, length_ratio_val,
        text_sim, skill_text_sim, embedding_sim,
        skill_string_score, fuzzy_score
    ]], columns=FEATURE_COLS)

    suitability = suitability_model.predict(features_df)[0]

    return {
        "category": category,
        "category_confidence": proba,
        "classes": classes,
        "order": order,
        "suitability_score": float(np.clip(suitability, 0, 100)),
        "matched_skills": sorted(resume_set & job_set),
    }


# ============ GLOBAL DARK THEME STYLING ============
st.markdown("""
<style>
:root {
    --bg-main: #0A0E1A;
    --bg-card: #131829;
    --bg-card-alt: #1A2036;
    --border: #262D45;
    --purple: #8B5CF6;
    --purple-dark: #6D28D9;
    --teal: #2DD4BF;
    --text-main: #E8EAF0;
    --text-muted: #8891A8;
}

.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
}

section[data-testid="stSidebar"] {
    background-color: #0D1220;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4, h5, p, span, div, label {
    color: var(--text-main);
}

.eyebrow {
    display: inline-block;
    color: var(--teal);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    font-weight: 600;
    text-transform: uppercase;
    background: rgba(45, 212, 191, 0.1);
    border: 1px solid rgba(45, 212, 191, 0.3);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

.hero {
    padding: 1rem 0 2rem 0;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 1rem;
    color: #FFFFFF;
}
.hero p {
    color: var(--text-muted);
    font-size: 1.05rem;
    max-width: 640px;
    line-height: 1.6;
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    margin-right: 0.4rem;
}
.badge-purple { background: rgba(139, 92, 246, 0.15); color: #C4B5FD; border: 1px solid rgba(139, 92, 246, 0.3); }
.badge-teal { background: rgba(45, 212, 191, 0.15); color: #5EEAD4; border: 1px solid rgba(45, 212, 191, 0.3); }
.badge-warn { background: rgba(245, 158, 11, 0.15); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.3); }

.result-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.12) 0%, rgba(45,212,191,0.08) 100%);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
}
.result-label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.result-value { font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0; }

.conf-row { margin-bottom: 0.9rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.3rem; }
.conf-track { background: #1E2438; border-radius: 8px; height: 10px; overflow: hidden; }
.conf-fill { background: linear-gradient(90deg, var(--purple), var(--teal)); height: 100%; border-radius: 8px; }

.stButton > button {
    background: linear-gradient(135deg, var(--purple), var(--purple-dark));
    color: white;
    border: none;
    font-weight: 700;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9D6FFF, var(--purple));
    color: white;
}

.stTextArea textarea, .stTextInput input {
    background: var(--bg-card-alt) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

[data-testid="stMetricValue"] { color: #FFFFFF; }
[data-testid="stMetricLabel"] { color: var(--text-muted); }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ============ SIDEBAR NAVIGATION ============
with st.sidebar:
    st.markdown("## 🧾 Resume AI")
    st.caption("Category + suitability screening")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍  Try It", "📊  Model Performance", "🧠  Model & Method"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Student project · Resume + job-posting datasets\nNot an official hiring tool")

# ============ TRY IT ============
if page == "🔍  Try It":
    st.markdown('<div class="eyebrow">● TWO-STAGE PIPELINE: CATEGORY + SUITABILITY</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
        <h1>Paste a resume and a job.<br>See where it lands.</h1>
        <p>Stage 1 predicts the resume's job category. Stage 2 scores how well it
        actually fits the specific job description you paste in, using a model
        trained on real resume-job match data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        resume_text = st.text_area(
            "Resume text", height=220,
            placeholder="Paste the resume text here..."
        )
    with col2:
        job_text = st.text_area(
            "Job description", height=220,
            placeholder="Paste the job description here..."
        )
    predict_clicked = st.button("🔎  Score This Resume", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        if resume_text.strip() == "" or job_text.strip() == "":
            st.warning("Paste both a resume and a job description first.")
        else:
            with st.spinner("Scoring..."):
                result = score_resume_against_job(resume_text, job_text)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Predicted Category</div>
                    <div class="result-value">{result['category']}</div>
                    <span class="badge badge-purple">Stage 1 · Logistic Regression</span>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                score = result["suitability_score"]
                badge_class = "badge-teal" if score >= 60 else "badge-warn"
                fit_label = "Strong fit" if score >= 60 else ("Moderate fit" if score >= 30 else "Weak fit")
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Suitability Score</div>
                    <div class="result-value">{score:.1f} / 100</div>
                    <span class="badge {badge_class}">{fit_label}</span>
                    <span class="badge badge-purple">Stage 2 · Gradient Boosting</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Category confidence breakdown")
            classes, order, proba = result["classes"], result["order"], result["category_confidence"]
            bars_html = '<div class="card">'
            for i in order[:5]:
                pct = proba[i] * 100
                bars_html += (
                    '<div class="conf-row">'
                    f'<div class="conf-label"><span>{classes[i]}</span><span>{pct:.1f}%</span></div>'
                    f'<div class="conf-track"><div class="conf-fill" style="width:{pct}%;"></div></div>'
                    '</div>'
                )
            bars_html += '</div>'
            st.markdown(bars_html, unsafe_allow_html=True)

            st.markdown("#### Matched skills")
            if result["matched_skills"]:
                skill_badges = "".join(
                    f'<span class="badge badge-teal">{s}</span>' for s in result["matched_skills"]
                )
                st.markdown(f'<div class="card">{skill_badges}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card">No overlapping skills detected between the resume and job description.</div>', unsafe_allow_html=True)

            st.caption(
                "Suitability score is a trained model estimate, not a hiring decision. "
                "Two of its inputs (skill_string_match_score, fuzzy_match_score) are this app's "
                "own approximations - see Model & Method for details."
            )

# ============ MODEL PERFORMANCE ============
elif page == "📊  Model Performance":
    st.markdown('<div class="eyebrow">● EVALUATED ON HELD-OUT TEST DATA</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>Test set performance</h1></div>', unsafe_allow_html=True)

    st.markdown("### Stage 1 - Category Classifier")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", "68.0%")
    with m2:
        st.metric("Categories", "24 → 20")
    with m3:
        st.metric("Model", "Logistic Regression")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Confusion matrix")
    st.image("chart_confusion_matrix.png", use_container_width=True)
    st.caption(
        "Most confusion happens between categories with genuine language overlap - "
        "e.g. Consultant, Business-Development, and Sales resumes were merged into one "
        "Business category during cleanup for exactly this reason."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Stage 2 - Suitability Scoring")
    m4, m5, m6 = st.columns(3)
    with m4:
        st.metric("MAE", "8.69")
    with m5:
        st.metric("R²", "0.814")
    with m6:
        st.metric("Model", "Gradient Boosting")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Predicted vs actual")
        st.image("chart_predicted_vs_actual.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Feature importance")
        st.image("chart_feature_importance.png", use_container_width=True)
        st.caption("Plain TF-IDF text similarity ended up the strongest signal - sentence embeddings added only marginal value on top of it.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============ MODEL & METHOD ============
else:
    st.markdown('<div class="eyebrow">● HOW IT WORKS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>Project overview</h1><p>A two-stage system: first predict what kind of resume this is, then score how well it fits a specific job description - built on a resume dataset too small to train reliably without deliberate handling of class imbalance and missing negative examples.</p></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Resumes", "2,484")
    with m2:
        st.metric("Categories", "24 → 20")
    with m3:
        st.metric("Stage 1 accuracy", "68.0%")
    with m4:
        st.metric("Stage 2 R²", "0.814")

    st.markdown("### The pipeline")
    p1, p2, p3, p4, p5 = st.columns(5)
    steps = [
        ("1. EDA & Clean", "Dropped duplicate/near-empty resumes, stripped stray HTML tags and whitespace from scraped text."),
        ("2. Merge categories", "Consultant, Business-Development, and Sales merged into one Business bucket - the language between them genuinely overlaps."),
        ("3. Stage 1 train", "TF-IDF (unigrams+bigrams) + Logistic Regression, class_weight='balanced' so small categories aren't ignored."),
        ("4. Stage 2 features", "8 features per resume-job pair: skill overlap, length ratio, TF-IDF similarity, skill-text similarity, sentence embeddings, plus two dataset-provided match scores."),
        ("5. Stage 2 train", "Synthetic negative pairs (wrong-category job matches) added so the model learns what a bad fit looks like, not just gradations of good ones."),
    ]
    for col, (title, desc) in zip([p1, p2, p3, p4, p5], steps):
        with col:
            st.markdown(f'<div class="card"><b>{title}</b><br><span style="color:#8891A8;font-size:0.85rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("### Key engineering decisions")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="card">
        <b>Why merge categories</b><br><br>
        The raw dataset has 24 categories, some with under 40 samples. Consultant,
        Business-Development, and Sales were constantly confused with each other in
        the confusion matrix - merging them and dropping the smallest classes
        (Automobile, BPO) lifted accuracy from 66% to 68% and made the remaining
        errors more meaningful to analyze.
        </div>
        <div class="card">
        <b>Why synthetic negative pairs</b><br><br>
        The suitability training data only pairs resumes with jobs they were
        actually matched against - every example is some flavor of "decent fit."
        Pairing some resumes with a random wrong-category job and labeling it with
        a low score gave the model real contrast to learn from, which measurably
        fixed the worst-performing categories (HR error dropped from 20.7 to ~11.5).
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="card">
        <b>Why Gradient Boosting over Random Forest</b><br><br>
        Both were tried alongside a tuned Random Forest (GridSearchCV). Gradient
        Boosting won on every metric (MAE 8.69 vs 8.98-9.04), so it became the
        final Stage 2 model.
        </div>
        <div class="card">
        <b>What didn't work as expected</b><br><br>
        Sentence embeddings were added expecting a meaningful semantic-similarity
        boost over plain TF-IDF. In practice, embedding_similarity contributed only
        ~6% feature importance vs ~47% for plain text_similarity - resume-job fit in
        this data is driven more by shared vocabulary than by paraphrased meaning.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Honest limitations")
    l1, l2, l3 = st.columns(3)
    limitations = [
        ("Approximated match scores", "skill_string_match_score and fuzzy_match_score in Stage 2 training came from the original dataset author's private RecAI pipeline. This app's live scoring uses honest, simpler stand-ins (word-boundary matching, difflib) - not the exact original formula."),
        ("Small training set", "2,484 resumes across 20+ categories means some categories (Agriculture, Apparel) only have a few dozen examples - Stage 1's 68% accuracy reflects a genuinely hard, data-limited problem, not a modeling mistake."),
        ("Category errors propagate", "If Stage 1 misclassifies a resume's category, that error can carry into how Stage 2's features are interpreted, even though Stage 2 doesn't take Stage 1's prediction as a direct input."),
    ]
    for col, (title, desc) in zip([l1, l2, l3], limitations):
        with col:
            st.markdown(f'<div class="card"><span class="badge badge-warn">{title}</span><br><br><span style="color:#8891A8;font-size:0.9rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Stage 1: scikit-learn LogisticRegression · Stage 2: GradientBoostingRegressor · random_state=42 · trained on Google Colab (free tier)")
