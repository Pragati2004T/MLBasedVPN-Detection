import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df.columns = df.columns.str.strip().str.lower()

print("\nDataset loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# -----------------------------------
# Validate required columns
# -----------------------------------
required_cols = ["proto", "service", "state", "sbytes", "dbytes", "attack_cat"]
missing_required = [col for col in required_cols if col not in df.columns]

if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# -----------------------------------
# Features and target
# -----------------------------------
X = df.drop(columns=["attack_cat"])

# Keep label out of attack classification features
if "label" in X.columns:
    X = X.drop(columns=["label"])

y = df["attack_cat"].astype(str).str.strip()

print("\nInput feature columns:")
print(X.columns.tolist())

print("\nUnique attack categories:")
print(sorted(y.unique()))

print("\nAttack category counts:")
print(y.value_counts())

# -----------------------------------
# Clean X
# -----------------------------------
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("unknown_feature").astype(str).str.strip().str.lower()
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

# One-hot encode categorical columns
X = pd.get_dummies(X)

print("\nShape after one-hot encoding:", X.shape)

# -----------------------------------
# Encode target labels
# -----------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded classes:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i} -> {class_name}")

# -----------------------------------
# Train-test split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# -----------------------------------
# Resample training set (test set is NEVER touched)
# -----------------------------------
# Step 1: Undersample Normal to 3x the next-largest class
# Step 2: SMOTE the minority attack classes up to a viable count
#
# This reduces Normal dominance (95% -> manageable) while giving
# rare classes like Worms, Shellcode, Backdoor enough samples.

print("\nTraining class distribution BEFORE resampling:")
train_class_names = label_encoder.inverse_transform(y_train)
before_dist = pd.Series(train_class_names).value_counts()
print(before_dist)

# Determine undersampling cap for majority class (Normal)
normal_idx = list(label_encoder.classes_).index("Normal")
non_normal_max = before_dist.drop("Normal", errors="ignore").max()
normal_cap = int(non_normal_max * 3)

# Build per-class sampling strategy for undersampling
under_strategy = {}
for cls_idx in range(len(label_encoder.classes_)):
    cls_name = label_encoder.classes_[cls_idx]
    cls_count = int((y_train == cls_idx).sum())
    if cls_name == "Normal":
        under_strategy[cls_idx] = min(cls_count, normal_cap)
    else:
        under_strategy[cls_idx] = cls_count  # keep all attack samples

print(f"\nUndersampling Normal from {int((y_train == normal_idx).sum()):,} to {under_strategy[normal_idx]:,}")

undersampler = RandomUnderSampler(
    sampling_strategy=under_strategy,
    random_state=42
)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

print(f"Training shape after undersampling: {X_train_under.shape}")

# SMOTE: bring all minority classes up to a minimum viable count
# Use the median class count as the SMOTE target (avoids over-inflating tiny classes)
post_under_dist = pd.Series(y_train_under).value_counts()
median_count = int(post_under_dist.median())
smote_min = max(median_count, 2000)  # at least 2000 per class

smote_strategy = {}
for cls_idx, count in post_under_dist.items():
    if count < smote_min:
        smote_strategy[cls_idx] = smote_min
    # classes already above smote_min are left as-is

if smote_strategy:
    # SMOTE needs k_neighbors <= smallest class count - 1
    smallest_minority = min(post_under_dist[cls] for cls in smote_strategy)
    k_neighbors = min(5, smallest_minority - 1)
    k_neighbors = max(1, k_neighbors)

    print(f"\nApplying SMOTE (target={smote_min:,} per minority class, k_neighbors={k_neighbors})")

    smoter = SMOTE(
        sampling_strategy=smote_strategy,
        random_state=42,
        k_neighbors=k_neighbors
    )
    X_train_resampled, y_train_resampled = smoter.fit_resample(X_train_under, y_train_under)
else:
    X_train_resampled, y_train_resampled = X_train_under, y_train_under

print(f"\nTraining shape after resampling: {X_train_resampled.shape}")
print("\nTraining class distribution AFTER resampling:")
resampled_names = label_encoder.inverse_transform(y_train_resampled)
print(pd.Series(resampled_names).value_counts())

# -----------------------------------
# Train Random Forest
# -----------------------------------
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    max_depth=30,
    min_samples_leaf=2
)

model.fit(X_train_resampled, y_train_resampled)

print("Training completed.")

# -----------------------------------
# Predict (on original untouched test set)
# -----------------------------------
y_pred = model.predict(X_test)

# -----------------------------------
# Evaluate
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 60)
print("EVALUATION ON ORIGINAL TEST SET (untouched, stratified)")
print("=" * 60)
print(f"\nAccuracy:        {accuracy:.4f}")
print(f"Macro Avg F1:    {macro_f1:.4f}")
print(f"Weighted Avg F1: {weighted_f1:.4f}")

print("\nPer-Class Classification Report:")
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
)
print(report)

# Per-class recall (the key metric for minority classes)
print("Per-Class Recall:")
cm = confusion_matrix(y_test, y_pred)
for i, cls_name in enumerate(label_encoder.classes_):
    if cm[i].sum() > 0:
        recall = cm[i, i] / cm[i].sum()
        print(f"  {cls_name:20s}  {recall:.4f}  ({cm[i, i]}/{cm[i].sum()})")

print("\nConfusion Matrix:")
print(cm)

# -----------------------------------
# Save model and helper files (v2 — new filenames for comparison)
# -----------------------------------
# The original v1 files are preserved so you can compare:
#   v1: ids_rf_model.joblib  (old, no resampling)
#   v2: ids_rf_v2_model.joblib  (new, with undersample + SMOTE)

joblib.dump(model, os.path.join(MODEL_DIR, "ids_rf_v2_model.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "ids_rf_v2_label_encoder.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "ids_rf_v2_feature_columns.joblib"))

# Save training distribution for concept drift
# Use the ORIGINAL training split distribution (not resampled) because
# concept drift compares against real-world expected class proportions.
train_dist = pd.Series(
    label_encoder.inverse_transform(y_train)
).value_counts(normalize=True).to_dict()

joblib.dump(train_dist, os.path.join(MODEL_DIR, "ids_rf_v2_training_distribution.joblib"))

print("\nv2 model and helper files saved successfully.")
print("Saved files (new — old v1 files are untouched):")
print("- models/ids_rf_v2_model.joblib")
print("- models/ids_rf_v2_label_encoder.joblib")
print("- models/ids_rf_v2_feature_columns.joblib")
print("- models/ids_rf_v2_training_distribution.joblib")