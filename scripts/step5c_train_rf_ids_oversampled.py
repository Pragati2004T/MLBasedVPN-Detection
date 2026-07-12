"""
step5c_train_rf_ids_oversampled.py
Experiment: IDS Random Forest with RandomOverSampler on minority classes.

This script trains a Random Forest classifier on the UNSW-NB15 dataset
using RandomOverSampler to handle class imbalance. Oversampling is applied
ONLY to the training set — the test set is never touched.

Output model files use the 'oversampled' prefix so existing models are untouched.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from imblearn.over_sampling import RandomOverSampler

# -----------------------------------
# Paths
# -----------------------------------
DATA_PATH = "data/processed/unsw_merged_cleaned.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------
# 1. Load dataset
# -----------------------------------
print("=" * 60)
print("IDS RF — RandomOverSampler Experiment")
print("=" * 60)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Normalize column names (strip whitespace, lowercase)
df.columns = df.columns.str.strip().str.lower()
print(f"Dataset shape: {df.shape}")

# -----------------------------------
# 2. Validate required columns
# -----------------------------------
required_cols = ["proto", "service", "state", "sbytes", "dbytes", "attack_cat"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -----------------------------------
# 3. Select features and target
# -----------------------------------
# Use only these 5 features (same as the production pipeline expects)
X = df[["proto", "service", "state", "sbytes", "dbytes"]].copy()

# Target: attack category (10 classes including Normal)
y = df["attack_cat"].astype(str).str.strip()

print(f"\nFeatures: {X.columns.tolist()}")
print(f"Target classes: {sorted(y.unique())}")
print(f"\nClass distribution in full dataset:")
print(y.value_counts().to_string())

# -----------------------------------
# 4. Clean features
# -----------------------------------
# Handle categorical columns: fill missing values, normalize text
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("unknown_feature").astype(str).str.strip().str.lower()
    else:
        # Convert to numeric, fill NaN with median
        X[col] = pd.to_numeric(X[col], errors="coerce")
        col_median = X[col].median()
        X[col] = X[col].fillna(col_median if pd.notna(col_median) else 0)

# One-hot encode categorical columns (proto, service, state)
X = pd.get_dummies(X)
print(f"\nShape after one-hot encoding: {X.shape}")

# -----------------------------------
# 5. Encode target labels
# -----------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nEncoded classes:")
for i, name in enumerate(label_encoder.classes_):
    print(f"  {i} -> {name}")

# -----------------------------------
# 6. Train-test split (BEFORE any resampling)
# -----------------------------------
# 80% train, 20% test, stratified to preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

# -----------------------------------
# 7. Apply RandomOverSampler (training set ONLY)
# -----------------------------------
# RandomOverSampler duplicates real minority samples until all classes
# have the same number of samples as the majority class.
# Unlike SMOTE, it does not create synthetic samples — it copies real ones.
print("\n" + "-" * 60)
print("RESAMPLING (training set only — test set is NEVER touched)")
print("-" * 60)

# Show distribution before resampling
train_names_before = label_encoder.inverse_transform(y_train)
before_dist = pd.Series(train_names_before).value_counts()
print(f"\nTraining distribution BEFORE oversampling:")
print(before_dist.to_string())

# Apply RandomOverSampler — balances all classes to match the largest class
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Show distribution after resampling
train_names_after = label_encoder.inverse_transform(y_train_resampled)
after_dist = pd.Series(train_names_after).value_counts()
print(f"\nTraining distribution AFTER oversampling:")
print(after_dist.to_string())
print(f"\nResampled training shape: {X_train_resampled.shape}")

# -----------------------------------
# 8. Train Random Forest
# -----------------------------------
print("\n" + "-" * 60)
print("TRAINING")
print("-" * 60)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    max_depth=30,
    min_samples_leaf=2,
)

print("Training Random Forest (300 trees, balanced, max_depth=30)...")
model.fit(X_train_resampled, y_train_resampled)
print("Training completed.")

# -----------------------------------
# 9. Evaluate on the original untouched test set
# -----------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 60)
print("EVALUATION — original test set (untouched, stratified 20%)")
print("=" * 60)
print(f"\n  Accuracy:         {accuracy:.4f}")
print(f"  Macro Avg F1:     {macro_f1:.4f}")
print(f"  Weighted Avg F1:  {weighted_f1:.4f}")

print("\nFull Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_,
    zero_division=0,
))

# Per-class recall (the key metric for minority classes)
print("Per-Class Recall:")
cm = confusion_matrix(y_test, y_pred)
for i, name in enumerate(label_encoder.classes_):
    total = cm[i].sum()
    if total > 0:
        recall = cm[i, i] / total
        print(f"  {name:20s}  {recall:.4f}  ({cm[i, i]:,} / {total:,})")

print(f"\nConfusion Matrix:\n{cm}")

# -----------------------------------
# 10. Save model artifacts (new filenames — existing models untouched)
# -----------------------------------
print("\n" + "-" * 60)
print("SAVING MODEL ARTIFACTS")
print("-" * 60)

joblib.dump(model, os.path.join(MODEL_DIR, "ids_rf_oversampled_model.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "ids_rf_oversampled_label_encoder.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "ids_rf_oversampled_feature_columns.joblib"))

# Save ORIGINAL training distribution (before resampling) for concept drift.
# Concept drift compares against real-world proportions, not resampled ones.
train_dist = pd.Series(
    label_encoder.inverse_transform(y_train)
).value_counts(normalize=True).to_dict()
joblib.dump(train_dist, os.path.join(MODEL_DIR, "ids_rf_oversampled_training_distribution.joblib"))

print("\nSaved (existing model files are untouched):")
print("  models/ids_rf_oversampled_model.joblib")
print("  models/ids_rf_oversampled_label_encoder.joblib")
print("  models/ids_rf_oversampled_feature_columns.joblib")
print("  models/ids_rf_oversampled_training_distribution.joblib")

print("\n" + "=" * 60)
print("Done. Compare these results against your baseline.")
print("Key metrics: macro F1 and per-class recall for")
print("Analysis, Backdoor, DoS, Shellcode, Worms.")
print("=" * 60)
