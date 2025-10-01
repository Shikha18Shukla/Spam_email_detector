# app/train_model.py
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

DATA_PATH = os.path.join("data", "emails.csv")
MODEL_PATH = os.path.join("models", "email_model.joblib")
RANDOM_STATE = 42

# ---------- Helpers ----------
def find_columns(df):
    """Try to detect text column and label column from common names."""
    text_candidates = ["text", "Text", "content", "Content", "message", "Message", "body", "Body"]
    label_candidates = ["label", "Label", "class", "Class", "target", "Target"]
    text_col = None
    label_col = None

    for c in df.columns:
        if c in text_candidates:
            text_col = c
            break
    for c in df.columns:
        if c in label_candidates:
            label_col = c
            break

    # fallback: if dataset has exactly two columns, assume first is text, second is label
    if text_col is None or label_col is None:
        if len(df.columns) == 2:
            text_col, label_col = df.columns[0], df.columns[1]

    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Found columns: {df.columns.tolist()}")

    return text_col, label_col

def normalize_labels(s):
    """Map labels to binary 1=spam, 0=ham"""
    s = s.astype(str).str.strip()
    # if numeric already (0/1)
    if pd.api.types.is_numeric_dtype(s) or s.dropna().apply(lambda x: str(x).isdigit()).all():
        return s.astype(int).map(lambda x: 1 if x == 1 else 0)
    s_lower = s.str.lower()
    return s_lower.map(lambda x: 1 if ("spam" in x or x in ['1','true','t','fraud','phish']) else 0)

def clean_text(text):
    """Basic cleaning: lower, remove urls, emails, html entities, non-alphanum (keep spaces)."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)                # URLs
    text = re.sub(r"\S+@\S+", " ", text)                        # emails
    text = re.sub(r"&[a-z]+;", " ", text)                       # html entities
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)                 # non-alphanumeric
    text = re.sub(r"\s+", " ", text)                            # collapse spaces
    return text.strip().lower()

# ---------- Main ----------
def load_and_prepare(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Put your CSV here (data/emails.csv).")
    # try a robust read
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines='skip')

    # detect columns
    text_col, label_col = find_columns(df)
    print(f"Detected text column: '{text_col}', label column: '{label_col}'")

    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df['label'] = normalize_labels(df['label'])
    # basic cleaning
    df['text'] = df['text'].apply(clean_text)
    # drop any empty text rows
    df = df[df['text'].str.strip() != ""]
    df = df.reset_index(drop=True)
    return df

def build_and_train(df):
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Candidate pipelines
    pipelines = {
        'nb': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.98)),
                        ('clf', MultinomialNB())]),
        'lr': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.98)),
                        ('clf', LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'))]),
        'svc': Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.98)),
                         ('clf', LinearSVC(class_weight='balanced', max_iter=20000))])
    }

    # Small param grids (fast)
    param_grids = {
        'nb': {
            'tfidf__min_df': [1, 2],
            'tfidf__ngram_range': [(1,1), (1,2)]
        },
        'lr': {
            'tfidf__min_df': [1, 2],
            'clf__C': [0.5, 1.0]
        },
        'svc': {
            'tfidf__min_df': [1, 2],
            'clf__C': [0.5, 1.0]
        }
    }

    best_models = {}
    for name in pipelines:
        print(f"\n=== Training & tuning: {name} ===")
        pipeline = pipelines[name]
        grid = GridSearchCV(pipeline, param_grids[name], cv=4, scoring='f1', n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        print(f"Best params ({name}):", grid.best_params_)
        best_models[name] = grid.best_estimator_

    # Evaluate all on test set and pick the best by f1 (spam class)
    results = {}
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n--- Results for {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(report)
        print("Confusion matrix (rows true, cols pred):")
        print(cm)
        results[name] = {
            'model': model,
            'accuracy': acc,
            'report': report
        }

    # choose best by accuracy (or you can change to recall/f1)
    best_name = max(results.items(), key=lambda kv: kv[1]['accuracy'])[0]
    best_model = results[best_name]['model']
    print(f"\n*** Selected best model: {best_name} (accuracy={results[best_name]['accuracy']:.4f}) ***")

    # retrain best_model on full dataset (optional but recommended)
    print("Retraining best model on full dataset...")
    best_model.fit(X, y)

    # ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model ({best_name}) to: {MODEL_PATH}")
    return best_name, results[best_name]

if __name__ == "__main__":
    print("Loading dataset...")
    df = load_and_prepare(DATA_PATH)
    print(f"Loaded {len(df)} cleaned rows.")
    best_name, best_info = build_and_train(df)
    print("Done.")
