"""FastAPI ML service for ReUse Campus.
Run from this folder: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from pathlib import Path
from typing import Any, Optional
import re
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "category_classifier.joblib"
CATALOG_PATH = MODEL_DIR / "recommendation_catalog.csv"
VECTORIZER_PATH = MODEL_DIR / "recommendation_vectorizer.joblib"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Missing {MODEL_PATH}. Run the notebook first or include the trained model artifact.")
if not VECTORIZER_PATH.exists():
    raise RuntimeError(f"Missing {VECTORIZER_PATH}. Run the notebook first or include the trained model artifact.")

category_model = joblib.load(MODEL_PATH)
recommendation_vectorizer = joblib.load(VECTORIZER_PATH)
if CATALOG_PATH.exists():
    demo_catalog = pd.read_csv(CATALOG_PATH).fillna("")
else:
    demo_catalog = pd.DataFrame(columns=["title", "description", "category", "listing_type", "condition", "location", "status"])

app = FastAPI(title="ReUse Campus ML API", version="1.0.0",
              description="Listing category prediction and content-based recommendations.")

class CategoryRequest(BaseModel):
    title: str = Field(default="", max_length=500)
    description: str = Field(default="", max_length=5000)

class RecommendationRequest(BaseModel):
    query: str = Field(default="", max_length=1000)
    category: Optional[str] = None
    listing_type: Optional[str] = None
    location: Optional[str] = None
    exclude_listing_ids: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)
    # For production, pass live listings from MongoDB, including each listing's actual id.
    items: Optional[list[dict[str, Any]]] = None


def normalise_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def get_field(item: dict, *names, default=""):
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    return default

@app.get("/health")
def health():
    return {"status": "ok", "service": "reuse-campus-ml"}

@app.post("/predict-category")
def predict_category(payload: CategoryRequest):
    text = f"{normalise_text(payload.title)}. {normalise_text(payload.description)}".strip(" .")
    if len(text) < 2:
        raise HTTPException(status_code=422, detail="Provide a title or description.")
    try:
        predicted = str(category_model.predict([text])[0])
        probabilities = category_model.predict_proba([text])[0]
        classes = category_model.classes_
        confidence = float(np.max(probabilities))
        ranked = sorted(zip(classes, probabilities), key=lambda x: float(x[1]), reverse=True)[:3]
        return {
            "predicted_category": predicted,
            "confidence": round(confidence, 4),
            "needs_review": confidence < 0.55,
            "top_predictions": [{"category": str(label), "confidence": round(float(prob), 4)} for label, prob in ranked]
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}")

@app.post("/recommendations")
def recommendations(payload: RecommendationRequest):
    source = payload.items if payload.items is not None else demo_catalog.to_dict(orient="records")
    if not source:
        return {"recommendations": [], "count": 0, "message": "No listings supplied."}

    excluded = {str(x) for x in payload.exclude_listing_ids}
    filtered = []
    for item in source:
        listing_id = str(get_field(item, "listingId", "listing_id", "_id", "id", "demo_row_id", default=""))
        status = normalise_text(get_field(item, "status")).lower()
        # If status exists, don't recommend listings that are explicitly unavailable.
        if status and status not in {"available", "active"}:
            continue
        if listing_id and listing_id in excluded:
            continue
        if payload.category and normalise_text(get_field(item, "category")).lower() != payload.category.strip().lower():
            continue
        if payload.listing_type and normalise_text(get_field(item, "listingType", "listing_type")).lower() != payload.listing_type.strip().lower():
            continue
        if payload.location and payload.location.strip().lower() not in normalise_text(get_field(item, "location")).lower():
            continue
        filtered.append(item)

    if not filtered:
        return {"recommendations": [], "count": 0, "message": "No listings match these filters."}

    def item_text(item):
        return " ".join([
            normalise_text(get_field(item, "title")),
            normalise_text(get_field(item, "description")),
            normalise_text(get_field(item, "category")),
            normalise_text(get_field(item, "listingType", "listing_type")),
            normalise_text(get_field(item, "condition")),
        ]).strip()

    query = normalise_text(payload.query)
    if not query:
        query = " ".join(x for x in [payload.category, payload.listing_type, payload.location] if x)
    if query:
        query_vector = recommendation_vectorizer.transform([query])
        item_matrix = recommendation_vectorizer.transform([item_text(item) for item in filtered])
        scores = cosine_similarity(query_vector, item_matrix).ravel()
    else:
        # Without a preference query, give deterministic results; prefer high-view items if that field exists.
        scores = np.array([float(get_field(item, "views", default=0) or 0) for item in filtered], dtype=float)
        if scores.max(initial=0) > 0:
            scores = scores / scores.max()

    order = np.argsort(scores)[::-1][:payload.limit]
    results = []
    for idx in order:
        item = filtered[int(idx)]
        listing_id = get_field(item, "listingId", "listing_id", "_id", "id", "demo_row_id", default=None)
        results.append({
            "listing_id": str(listing_id) if listing_id is not None and str(listing_id) else None,
            "title": normalise_text(get_field(item, "title")),
            "category": normalise_text(get_field(item, "category")),
            "listing_type": normalise_text(get_field(item, "listingType", "listing_type")),
            "condition": normalise_text(get_field(item, "condition")),
            "location": normalise_text(get_field(item, "location")),
            "status": normalise_text(get_field(item, "status")),
            "score": round(float(scores[int(idx)]), 4),
        })
    return {"recommendations": results, "count": len(results), "source": "live_database" if payload.items is not None else "demo_catalog"}
