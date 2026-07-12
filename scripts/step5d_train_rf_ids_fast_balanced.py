import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# -----------------------------------
# Paths
# -----------------------------------
DATA_PATH = "data/processed/unsw_merged_cleaned.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("IDS RF — FAST BALANCED EXPERIMENT")
print("=" * 60)

# -----------------------------------
# Load dataset
# -----------------------------------
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

print("Dataset shape:", df.shape)

required_cols = ["proto", "service", "state", "sbytes", "dbytes", "attack_cat"]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -----------------------------------
# Features and target
# -----------------------------------
X = df.drop(columns=["attack_cat"])
if "label" in X.columns:
    X = X.drop(columns=["label"])

y = df["attack_cat"].astype(str).str.strip()

print("\nFeatures:", X.columns.tolist())
print("Target classes:", sorted(y.unique()))

print("\nClass distribution in full dataset:")
print(y.value_counts())

# -----------------------------------
# Clean features
# -----------------------------------
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("unknown_feature").astype(str).str.strip().str.lower()
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

# One-hot encode
X = pd.get_dummies(X)
print("\nShape after one-hot encoding:", X.shape)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded classes:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} -> {cls}")

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

print("\nTrain shape:", X_train.shape)
print("Test shape: ", X_test.shape)

print("\nTraining distribution BEFORE balancing:")
before_counts = pd.Series(label_encoder.inverse_transform(y_train)).value_counts()
print(before_counts)

# -----------------------------------
# Step 1: Undersample dominant class only
# -----------------------------------
# Find encoded value for Normal
normal_class_id = label_encoder.transform(["Normal"])[0]

train_counter = Counter(y_train)

# Keep Normal at a capped size, keep all others initially
undersample_strategy = {}
for cls_id, count in train_counter.items():
    if cls_id == normal_class_id:
        undersample_strategy[cls_id] = min(count, 250000)  # cap Normal
    else:
        undersample_strategy[cls_id] = count

rus = RandomUnderSampler(
    sampling_strategy=undersample_strategy,
    random_state=42
)

X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print("\nDistribution AFTER undersampling Normal:")
under_counts = pd.Series(label_encoder.inverse_transform(y_train_under)).value_counts()
print(under_counts)

# -----------------------------------
# Step 2: Oversample minority classes to a reasonable cap
# -----------------------------------
# We do NOT oversample to Normal size.
# We cap minority classes to avoid huge training set.
target_cap = 30000

under_counter = Counter(y_train_under)
oversample_strategy = {}

for cls_id, count in under_counter.items():
    if cls_id != normal_class_id and count < target_cap:
        oversample_strategy[cls_id] = target_cap

ros = RandomOverSampler(
    sampling_strategy=oversample_strategy,
    random_state=42
)

X_train_balanced, y_train_balanced = ros.fit_resample(X_train_under, y_train_under)

print("\nDistribution AFTER capped oversampling:")
balanced_counts = pd.Series(label_encoder.inverse_transform(y_train_balanced)).value_counts()
print(balanced_counts)

print("\nBalanced training shape:", X_train_balanced.shape)

# -----------------------------------
# Train Random Forest
# -----------------------------------
print("\n" + "-" * 60)
print("TRAINING")
print("-" * 60)

print("Training Random Forest (120 trees, max_depth=20)...")

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

model.fit(X_train_balanced, y_train_balanced)

print("Training completed.")

# -----------------------------------
# Predict
# -----------------------------------
y_pred = model.predict(X_test)

# -----------------------------------
# Evaluate
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0,
    output_dict=True
)

print("\nRandom Forest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\nMacro Avg F1:", report["macro avg"]["f1-score"])
print("Weighted Avg F1:", report["weighted avg"]["f1-score"])

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# Save model and helper files
# -----------------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "ids_rf_fast_balanced_model.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "ids_rf_fast_balanced_label_encoder.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "ids_rf_fast_balanced_feature_columns.joblib"))

train_dist = pd.Series(
    label_encoder.inverse_transform(y_train_balanced)
).value_counts(normalize=True).to_dict()

joblib.dump(train_dist, os.path.join(MODEL_DIR, "ids_rf_fast_balanced_training_distribution.joblib"))

print("\nSaved files:")
print("- models/ids_rf_fast_balanced_model.joblib")
print("- models/ids_rf_fast_balanced_label_encoder.joblib")
print("- models/ids_rf_fast_balanced_feature_columns.joblib")
print("- models/ids_rf_fast_balanced_training_distribution.joblib")