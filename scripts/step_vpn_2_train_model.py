import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------
# Paths
# -----------------------------------
DATA_PATH = "data/processed/vpn_merged.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------
# Load merged VPN dataset
# -----------------------------------
print("Loading VPN merged dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df.columns = df.columns.str.strip().str.lower()

print("\nDataset loaded successfully.")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# -----------------------------------
# Identify target column
# -----------------------------------
target_col = "class1"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

# -----------------------------------
# Separate features and target
# -----------------------------------
X = df.drop(columns=[target_col])
y = df[target_col].astype(str).str.strip()

print("\nTarget value counts:")
print(y.value_counts())

# -----------------------------------
# Handle missing values
# -----------------------------------
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("unknown").astype(str).str.strip().str.lower()
    else:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

# -----------------------------------
# One-hot encode categorical columns if any
# -----------------------------------
X = pd.get_dummies(X)

print("\nShape after encoding:", X.shape)

# -----------------------------------
# Encode target labels
# -----------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded target classes:")
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
# Train Random Forest model
# -----------------------------------
print("\nTraining VPN Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
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

print("\nVPN Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# Save model and helper files
# -----------------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "vpn_rf_model.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "vpn_label_encoder.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "vpn_feature_columns.joblib"))

print("\nVPN model and helper files saved successfully.")
print("Saved files:")
print("- models/vpn_rf_model.joblib")
print("- models/vpn_label_encoder.joblib")
print("- models/vpn_feature_columns.joblib")