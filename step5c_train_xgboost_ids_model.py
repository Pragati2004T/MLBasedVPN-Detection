import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

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
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

print("\nColumns:")
print(df.columns.tolist())

# -----------------------------------
# Check target column
# -----------------------------------
if "attack_cat" not in df.columns:
    raise ValueError("Target column 'attack_cat' not found in dataset.")

# -----------------------------------
# Define features and target
# -----------------------------------
X = df.drop(columns=["attack_cat"])

if "label" in X.columns:
    X = X.drop(columns=["label"])

y = df["attack_cat"].astype(str).str.strip()

print("\nInput feature columns:")
print(X.columns.tolist())

print("\nUnique attack categories:")
print(sorted(y.unique()))

# -----------------------------------
# One-hot encode categorical columns
# -----------------------------------
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
# Train XGBoost model
# -----------------------------------
print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

print("Training completed.")

# -----------------------------------
# Predict
# -----------------------------------
y_pred = model.predict(X_test)

# -----------------------------------
# Evaluate
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nXGBoost Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# -----------------------------------
# Save model and helper files
# -----------------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "ids_xgboost_model.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "ids_xgboost_label_encoder.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "ids_xgboost_feature_columns.joblib"))

print("\nXGBoost model and helper files saved successfully.")
print("Saved files:")
print("- models/ids_xgboost_model.joblib")
print("- models/ids_xgboost_label_encoder.joblib")
print("- models/ids_xgboost_feature_columns.joblib")