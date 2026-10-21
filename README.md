# ReUse Campus — ML Module

This package provides the ML components suggested by the project proposal: listing category classification and content-based recommendations.

## Included

- `ReUse_Campus_ML.ipynb` — end-to-end notebook: load data, normalise category labels, train/evaluate classifier, confusion matrix, recommendation component, and optional request-volume plot.
- `models/category_classifier.joblib` — trained TF-IDF + Logistic Regression pipeline.
- `models/category_metrics.json` — evaluation metrics from a stratified holdout test.
- `models/recommendation_vectorizer.joblib` — fitted TF-IDF vectorizer for recommendation text.
- `models/recommendation_catalog.csv` — demo catalogue. Its `DEMO-...` IDs are not real database listing IDs.
- `app.py` — FastAPI endpoints for backend integration.
- `requirements.txt` — dependencies.

## Model choice

1. **Category classifier:** TF-IDF word and character n-grams + Logistic Regression. It predicts one of: `books`, `calculator`, `hostel essentials`, `lab equipment`, `others`, `stationery`.
2. **Recommendation system:** TF-IDF + cosine similarity over title, description, category, listing type and condition. This is a content-based recommendation component, not a collaborative filtering model.

A fraud or late-return classifier is intentionally not included: the provided data does not contain a reliable ground-truth fraud label, and late return must not be equated with fraud. Category-specific demand forecasting is also not included because the supplied cleaned listings CSV has no `listing_id`, preventing a reliable join between requests and listing categories. The notebook can plot overall request volume if the requests CSV is present.

## Run the notebook

Open `ReUse_Campus_ML.ipynb` in Jupyter or Google Colab. The `data/` folder contains the supplied listing and request CSVs. Run cells from top to bottom. The notebook writes model artifacts to `models/`.

## Run the ML API locally

From this folder, in a terminal:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Health check: `GET http://127.0.0.1:8000/health`

### Predict category

`POST http://127.0.0.1:8000/predict-category`

```json
{
  "title": "Casio scientific calculator",
  "description": "Working condition, useful for engineering mathematics."
}
```

Example response structure:

```json
{
  "predicted_category": "calculator",
  "confidence": 0.91,
  "needs_review": false,
  "top_predictions": [
    {"category": "calculator", "confidence": 0.91}
  ]
}
```

Confidence values are estimates, not guarantees. If `needs_review` is true, let the student select/confirm the category rather than automatically saving it.

### Recommendations

`POST http://127.0.0.1:8000/recommendations`

For production, the Express backend should query current listings from MongoDB and pass them in `items`. Use the real Mongo `_id`/listing ID in every item. Example:

```json
{
  "query": "engineering calculator for semester",
  "category": "calculator",
  "listing_type": "lend",
  "location": "Hostel A",
  "limit": 5,
  "exclude_listing_ids": [],
  "items": [
    {
      "_id": "REAL_MONGODB_LISTING_ID",
      "title": "Casio scientific calculator",
      "description": "Good condition",
      "category": "calculator",
      "listingType": "lend",
      "condition": "good",
      "location": "Hostel A",
      "status": "available"
    }
  ]
}
```

The endpoint returns the real listing IDs, so the frontend can link recommendations to listing detail pages. If `items` is omitted, the API uses the demo catalogue; those IDs are only demo IDs and must not be used to fetch production listings.

## Express backend example

```js
const response = await fetch("http://127.0.0.1:8000/predict-category", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ title, description })
});
const prediction = await response.json();
// If prediction.needs_review is true, ask the user to confirm the category.
```

For recommendations, query available listings from MongoDB first, then send those objects to `/recommendations`. Do not expose the ML API publicly without rate limits and request validation.

## Important integration notes

- Keep the database/backend as the source of truth for listing availability and permissions; ML must not accept/approve requests or change status.
- The model normalises `stationary` to `stationery`, and `hostel essentail`/`hostel essential` to `hostel essentials`. Align these labels with your backend's allowed category values or map them before saving.
- The provided cleaned listings file has no `listing_id`; it is not possible to reliably link its rows to `requests_raw(1).csv` by listing. Production recommendation requests must use live database records with real IDs.
- The model was evaluated on a random stratified holdout split from the supplied dataset. Validate with new real campus listings before making important decisions.
