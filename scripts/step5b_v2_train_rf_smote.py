"""
step5b_v2_train_rf_smote.py
Experiment: IDS Random Forest with undersample-Normal + SMOTE-minorities.

Differences from baseline (step5b_train_rf_ids_model.py):
  - Normal undersampled to 3x next-largest class (training set only)
  - SMOTE applied to minority classes (training set only)
  - 300 trees, class_weight='balanced', max_depth=30, min_samples_leaf=2
  - Saves model artifacts with 'v2' prefix so baseline is untouched
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# -----------------------------------
# Paths
# -----------------------------------
DATA_PATH = "data/processed/unsw_merged_cleaned.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------
# Load dataset
# -----------------------------------
print("=" * 60)
print("IDS RF v2 — Undersample + SMOTE experiment")
print("=" * 60)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

print(f"Dataset shape: {df.shape}")

# -----------------------------------
# Validate required columns
# -----------------------------------
required_cols = ["proto", "service", "state", "sbytes", "dbytes", "attack_cat"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -----------------------------------
# Features and target
# -----------------------------------
X = df[["proto", "service", "state", "sbytes", "dbytes"]].copy()
y = df["attack_cat"].astype(str).str.strip()

print(f"\nFeatures: {X.columns.tolist()}")
print(f"Target classes: {sorted(y.unique())}")
print(f"\nClass distribution in full dataset:")
print(y.value_counts().to_string())

# -----------------------------------
# Clean features
# -----------------------------------
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("unknown_feature").astype(str).str.strip().str.lower()
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

X = pd.get_dummies(X)
print(f"\nShape after one-hot encoding: {X.shape}")

# -----------------------------------
# Encode target
# -----------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nEncoded classes:")
for i, name in enumerate(label_encoder.classes_):
    print(f"  {i} -> {name}")

# -----------------------------------
# Train-test split (identical to baseline: 80/20, random_state=42)
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

# -----------------------------------
# STEP A: Undersample Normal in training set only
# -----------------------------------
print("\n" + "-" * 60)
print("RESAMPLING (training set only — test set is NEVER touched)")
print("-" * 60)

train_names = label_encoder.inverse_transform(y_train)
before_dist = pd.Series(train_names).value_counts()
print(f"\nTraining distribution BEFORE resampling:")
print(before_dist.to_string())

normal_idx = list(label_encoder.classes_).index("Normal")
non_normal_max = before_dist.drop("Normal", errors="ignore").max()
normal_cap = int(non_normal_max * 3)

under_strategy = {}
for idx in range(len(label_encoder.classes_)):
    name = label_encoder.classes_[idx]
    count = int((y_train == idx).sum())
    if name == "Normal":
        under_strategy[idx] = min(count, normal_cap)
    else:
        under_strategy[idx] = count  # keep every real attack sample

print(f"\nUndersampling Normal: {int((y_train == normal_idx).sum()):,} -> {under_strategy[normal_idx]:,}")

undersampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
X_under, y_under = undersampler.fit_resample(X_train, y_train)

print(f"Shape after undersampling: {X_under.shape}")

# -----------------------------------
# STEP B: SMOTE minority attack classes in training set only
# -----------------------------------
post_dist = pd.Series(y_under).value_counts()
median_count = int(post_dist.median())
smote_floor = max(median_count, 2000)

smote_strategy = {}
for idx, count in post_dist.items():
    if count < smote_floor:
        smote_strategy[idx] = smote_floor

if smote_strategy:
    smallest = min(post_dist[idx] for idx in smote_strategy)
    k = min(5, smallest - 1)
    k = max(1, k)

    print(f"\nSMOTE: boosting {len(smote_strategy)} classes to {smote_floor:,} samples (k_neighbors={k})")
    for idx in smote_strategy:
        name = label_encoder.classes_[idx]
        print(f"  {name}: {int(post_dist[idx]):,} -> {smote_floor:,}")

    smoter = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=k)
    X_resampled, y_resampled = smoter.fit_resample(X_under, y_under)
else:
    X_resampled, y_resampled = X_under, y_under

print(f"\nFinal training shape: {X_resampled.shape}")
print(f"\nTraining distribution AFTER resampling:")
after_names = label_encoder.inverse_transform(y_resampled)
print(pd.Series(after_names).value_counts().to_string())

# -----------------------------------
# Train Random Forest
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
model.fit(X_resampled, y_resampled)
print("Training completed.")

# -----------------------------------
# Evaluate on original untouched test set
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

print("Per-Class Recall:")
cm = confusion_matrix(y_test, y_pred)
for i, name in enumerate(label_encoder.classes_):
    total = cm[i].sum()
    if total > 0:
        recall = cm[i, i] / total
        print(f"  {name:20s}  {recall:.4f}  ({cm[i, i]:,} / {total:,})")

print(f"\nConfusion Matrix:\n{cm}")

# -----------------------------------
# Save v2 model artifacts (baseline files untouched)
# -----------------------------------
print("\n" + "-" * 60)
print("SAVING v2 ARTIFACTS")
print("-" * 60)

joblib.dump(model, os.path.join(MODEL_DIR, "ids_rf_v2_model.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "ids_rf_v2_label_encoder.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "ids_rf_v2_feature_columns.joblib"))

# Concept drift: use ORIGINAL training distribution (real-world proportions)
train_dist = pd.Series(
    label_encoder.inverse_transform(y_train)
).value_counts(normalize=True).to_dict()
joblib.dump(train_dist, os.path.join(MODEL_DIR, "ids_rf_v2_training_distribution.joblib"))

print("\nSaved (baseline v1 files are untouched):")
print("  models/ids_rf_v2_model.joblib")
print("  models/ids_rf_v2_label_encoder.joblib")
print("  models/ids_rf_v2_feature_columns.joblib")
print("  models/ids_rf_v2_training_distribution.joblib")

print("\nDone. Compare these results against baseline (~97.66% accuracy).")
print("Key metrics to compare: macro F1 and per-class recall for")
print("Analysis, Backdoor, DoS, Shellcode, Worms.")
