from pathlib import Path
import re, json, shutil, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'cleaned_listing_raw.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

# Read cleaned listings supplied by the user.
df = pd.read_csv(DATA)
required = {'title', 'description', 'category'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f'Missing required columns: {sorted(missing)}')

def normalise_category(x):
    x = str(x).strip().lower()
    x = re.sub(r'\s+', ' ', x)
    aliases = {
        'hostel essentail': 'hostel essentials',
        'hostel essential': 'hostel essentials',
        'hostel essentials': 'hostel essentials',
        'stationery': 'stationery',
        'stationary': 'stationery',
        'calculators': 'calculator',
        'other': 'others',
    }
    return aliases.get(x, x)

df['category_clean'] = df['category'].map(normalise_category)
df['title_clean'] = df['title'].fillna('').astype(str).str.strip()
df['description_clean'] = df['description'].fillna('').astype(str).str.strip()
df['text'] = (df['title_clean'] + '. ' + df['description_clean']).str.strip()
# Do not train on rows without usable text or target.
train_df = df[(df['text'].str.len() > 2) & (df['category_clean'].str.len() > 0)].copy()
X = train_df['text']
y = train_df['category_clean']
if y.nunique() < 2:
    raise ValueError('At least two category classes are required.')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = Pipeline([
    ('features', FeatureUnion([
        ('word_tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                                      strip_accents='unicode', sublinear_tf=True)),
        ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                                      max_features=30000, sublinear_tf=True)),
    ])),
    ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced'))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
report = classification_report(y_test, pred, output_dict=True, zero_division=0)
labels = sorted(y.unique().tolist())
cm = confusion_matrix(y_test, pred, labels=labels).tolist()
joblib.dump(model, MODELS / 'category_classifier.joblib')
metrics = {
    'model': 'TF-IDF (word + character n-grams) + Logistic Regression',
    'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)),
    'total_rows_used': int(len(train_df)), 'accuracy': float(accuracy),
    'labels': labels, 'classification_report': report, 'confusion_matrix': cm,
    'note': 'Evaluation uses a random stratified split. Results are dataset-dependent and should be rechecked on real campus listings.'
}
(MODELS / 'category_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

# Build recommendation catalogue with cleaned categories and stable row key for notebook demo.
# Actual backend integration should send live MongoDB listings and their real IDs to the API.
rec_cols = [c for c in ['title_clean','description_clean','category_clean','listing_type','condition','location','status','original_price_inr','photos_count','views','posted_date'] if c in df.columns]
rec = df[rec_cols].copy()
rec = rec.rename(columns={'title_clean':'title','description_clean':'description','category_clean':'category'})
for c in ['title','description','category','listing_type','condition','location','status']:
    if c in rec.columns:
        rec[c] = rec[c].fillna('').astype(str).str.strip()
rec['demo_row_id'] = [f'DEMO-{i:04d}' for i in range(len(rec))]
rec['demo_row_id'] = [f'DEMO-{i:04d}' for i in range(len(rec))]
rec.to_csv(MODELS / 'recommendation_catalog.csv', index=False)

def listing_document(row):
    return ' '.join([str(row.get('title', '')), str(row.get('description', '')), str(row.get('category', '')), str(row.get('listing_type', '')), str(row.get('condition', ''))]).strip()

rec_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=40000, strip_accents='unicode')
rec_vectorizer.fit(rec.apply(listing_document, axis=1).tolist())
joblib.dump(rec_vectorizer, MODELS / 'recommendation_vectorizer.joblib')

# Save the category mapping and schema details for backend integration.
(MODELS / 'model_info.json').write_text(json.dumps({
    'category_model_file':'category_classifier.joblib',
    'input_fields':['title','description'],
    'output_field':'predicted_category',
    'categories':labels,
    'recommendation_method':'TF-IDF cosine similarity over title + description, with optional category/listing_type/location filters',
    'important':'Never use user IDs, owner IDs, emails, names, or other direct identifiers as model features.'
}, indent=2), encoding='utf-8')

print(json.dumps({'accuracy':accuracy,'train_rows':len(X_train),'test_rows':len(X_test),'labels':labels}, indent=2))
