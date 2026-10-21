from pathlib import Path
import nbformat as nbf
root=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook()
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md('''# ReUse Campus — ML Pipeline

**Project scope:** college/hostel resource-sharing platform for lending, donation and exchange.

This notebook trains two practical components:
1. **Listing category classifier** — TF-IDF (word + character n-grams) + Logistic Regression.
2. **Content-based recommendation system** — TF-IDF + cosine similarity.

The proposal describes ML as optional and suggests a category classifier or recommendations. The core request/lending workflow must continue to work even if the ML service is down.

**Important data notes:** The supplied cleaned listings file does not contain a `listing_id`, so the recommendation demo uses a generated `demo_row_id` only for notebook examples. In production, Express must pass live MongoDB listings with their actual `_id`/listing ID to the ML API. User names, emails, owner IDs and other direct identifiers are not model features. This dataset is a supplied project dataset; test results may not represent real-world performance.''')
md('## 1. Imports and paths')
code('''from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_STATE = 42
BASE_DIR = Path.cwd()
DATA_PATH = BASE_DIR / "data" / "cleaned_listing_raw.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "cleaned_listing_raw.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Data path:", DATA_PATH)
print("Model folder:", MODEL_DIR)''')
md('## 2. Load the cleaned listings')
code('''df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.head()''')
md('## 3. Prepare text and standardise category labels')
code('''required = {"title", "description", "category"}
missing_columns = required - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

def normalise_category(value):
    value = re.sub(r"\\s+", " ", str(value).strip().lower())
    aliases = {
        "hostel essentail": "hostel essentials",
        "hostel essential": "hostel essentials",
        "hostel essentials": "hostel essentials",
        "stationary": "stationery",
        "stationery": "stationery",
        "other": "others",
        "calculators": "calculator",
    }
    return aliases.get(value, value)

model_df = df.copy()
model_df["category_clean"] = model_df["category"].map(normalise_category)
model_df["title_clean"] = model_df["title"].fillna("").astype(str).str.strip()
model_df["description_clean"] = model_df["description"].fillna("").astype(str).str.strip()
model_df["text"] = (model_df["title_clean"] + ". " + model_df["description_clean"]).str.strip()
model_df = model_df[(model_df["text"].str.len() > 2) & (model_df["category_clean"].str.len() > 0)].copy()

print("Rows available for model:", len(model_df))
print("Category counts:")
display(model_df["category_clean"].value_counts())''')
md('## 4. Train/test split')
code('''X = model_df["text"]
y = model_df["category_clean"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))''')
md('## 5. Train the category classifier')
code('''category_model = Pipeline([
    ("features", FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=30000,
            strip_accents="unicode", sublinear_tf=True
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=30000, sublinear_tf=True
        )),
    ])),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
category_model.fit(X_train, y_train)
y_pred = category_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.3f}")
print("\\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))''')
md('## 6. Confusion matrix')
code('''labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Category Classifier — Confusion Matrix")
plt.xlabel("Predicted category")
plt.ylabel("Actual category")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()''')
md('## 7. Try one prediction')
code('''def predict_listing_category(title, description=""):
    text = f"{str(title or '').strip()}. {str(description or '').strip()}".strip(" .")
    if len(text) < 2:
        raise ValueError("Enter a title or description.")
    probabilities = category_model.predict_proba([text])[0]
    classes = category_model.classes_
    best_index = int(np.argmax(probabilities))
    return {
        "predicted_category": str(classes[best_index]),
        "confidence": round(float(probabilities[best_index]), 4),
        "needs_review": bool(probabilities[best_index] < 0.55)
    }

predict_listing_category(
    "Casio scientific calculator",
    "Working condition, useful for engineering mathematics."
)''')
md('## 8. Save the trained model and metrics')
code('''joblib.dump(category_model, MODEL_DIR / "category_classifier.joblib")
metrics = {
    "model": "TF-IDF word + character n-grams with Logistic Regression",
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "accuracy": float(accuracy),
    "labels": labels,
    "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    "confusion_matrix": cm.tolist(),
    "note": "Random stratified holdout evaluation; validate on real campus data before relying on predictions."
}
(MODEL_DIR / "category_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("Saved category model and metrics to:", MODEL_DIR)''')
md('## 9. Prepare the recommendation catalogue')
code('''rec = model_df.copy()
rec_cols = [c for c in ["title_clean", "description_clean", "category_clean", "listing_type",
                         "condition", "location", "status", "original_price_inr",
                         "photos_count", "views", "posted_date"] if c in rec.columns]
rec = rec[rec_cols].copy().rename(columns={
    "title_clean": "title", "description_clean": "description", "category_clean": "category"
})
for col in ["title", "description", "category", "listing_type", "condition", "location", "status"]:
    if col in rec:
        rec[col] = rec[col].fillna("").astype(str).str.strip()
rec = rec.reset_index(drop=True)
rec["demo_row_id"] = [f"DEMO-{i:04d}" for i in range(len(rec))]
rec.head()''')
md('## 10. Fit TF-IDF for content-based recommendations')
code('''def listing_document(row):
    return " ".join([
        str(row.get("title", "")),
        str(row.get("description", "")),
        str(row.get("category", "")),
        str(row.get("listing_type", "")),
        str(row.get("condition", "")),
    ]).strip()

recommendation_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=1, max_features=40000, strip_accents="unicode"
)
recommendation_documents = rec.apply(listing_document, axis=1).tolist()
recommendation_matrix = recommendation_vectorizer.fit_transform(recommendation_documents)
joblib.dump(recommendation_vectorizer, MODEL_DIR / "recommendation_vectorizer.joblib")
rec.to_csv(MODEL_DIR / "recommendation_catalog.csv", index=False)
print("Recommendation catalogue rows:", len(rec))
print("TF-IDF matrix shape:", recommendation_matrix.shape)''')
md('## 11. Recommendation function')
code('''def recommend_items(query="", category=None, listing_type=None, location=None, limit=5):
    candidates = rec.copy()
    if category:
        candidates = candidates[candidates["category"].str.lower() == category.strip().lower()]
    if listing_type:
        candidates = candidates[candidates["listing_type"].str.lower() == listing_type.strip().lower()]
    if location:
        candidates = candidates[candidates["location"].str.lower().str.contains(location.strip().lower(), na=False)]
    if "status" in candidates:
        # Demo only: do not show listings explicitly marked unavailable.
        candidates = candidates[candidates["status"].str.lower().isin(["available", "active", "requested", ""]) ]
    if candidates.empty:
        return candidates
    if query.strip():
        q_vec = recommendation_vectorizer.transform([query])
        item_vecs = recommendation_vectorizer.transform(candidates.apply(listing_document, axis=1).tolist())
        scores = cosine_similarity(q_vec, item_vecs).ravel()
    else:
        views = pd.to_numeric(candidates.get("views", 0), errors="coerce").fillna(0)
        scores = views.to_numpy(dtype=float)
    candidates = candidates.copy()
    candidates["similarity_score"] = scores
    return candidates.sort_values("similarity_score", ascending=False).head(limit)[
        [c for c in ["demo_row_id", "title", "category", "listing_type", "condition", "location", "status", "similarity_score"] if c in candidates]
    ]

recommend_items(query="engineering calculator for semester", limit=5)''')
md('## 12. Optional: inspect request activity (not a fraud model)')
code('''# Requests can support descriptive activity trends. The supplied listings CSV has no listing_id,
# so category-level demand cannot safely be joined to requests using these supplied files.
REQUESTS_PATH = BASE_DIR / "data" / "requests_raw.csv"
if not REQUESTS_PATH.exists():
    REQUESTS_PATH = BASE_DIR / "requests_raw(1).csv"
if REQUESTS_PATH.exists():
    requests_df = pd.read_csv(REQUESTS_PATH)
    if "requested_at" in requests_df.columns:
        requests_df["requested_at_parsed"] = pd.to_datetime(
            requests_df["requested_at"], errors="coerce", dayfirst=True
        )
        daily_requests = (requests_df.dropna(subset=["requested_at_parsed"])
                          .set_index("requested_at_parsed").resample("W").size())
        plt.figure(figsize=(12, 4))
        daily_requests.plot()
        plt.title("Weekly Request Volume (Descriptive Trend)")
        plt.xlabel("Week")
        plt.ylabel("Number of requests")
        plt.tight_layout()
        plt.show()
else:
    print("Optional requests CSV not found. Skipping this section.")''')
md('''## 13. What to tell the backend teammate

- Run the FastAPI service in this folder: `uvicorn app:app --reload --port 8000`.
- Health check: `GET http://127.0.0.1:8000/health`.
- Category prediction: `POST /predict-category` with JSON `{"title":"Casio calculator","description":"Scientific calculator in good condition"}`.
- Recommendations: `POST /recommendations` with JSON such as `{"query":"engineering calculator","category":"calculator","limit":5,"items":[...]}`. For production, Express should fetch current available listings from MongoDB and pass them in `items`, including the real listing `_id`/ID. The API returns those IDs so the frontend can open the actual listing.
- Install dependencies using `pip install -r requirements.txt`.

**Limitations:** category labels are standardised to `stationery` and `hostel essentials`; make sure the backend/database enum uses the same labels or map them at the API boundary. The supplied listing CSV has no listing ID, so the notebook's `DEMO-...` IDs are not production IDs. Do not train a fraud/late-return model from these files: there is no trustworthy ground-truth fraud label, and an overdue/late return is not automatically fraud.''')
nb['cells']=cells
nb['metadata']={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.x'}}
nbf.write(nb, root/'ReUse_Campus_ML.ipynb')
