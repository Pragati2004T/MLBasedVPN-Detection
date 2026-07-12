import os
import joblib
import pandas as pd

# -----------------------------------
# Paths
# -----------------------------------
MODEL_PATH = "models/vpn_rf_model.joblib"
LABEL_ENCODER_PATH = "models/vpn_label_encoder.joblib"
FEATURE_COLUMNS_PATH = "models/vpn_feature_columns.joblib"

INPUT_FILE_PATH = "data/processed/vpn_merged.csv"

# -----------------------------------
# Load model files
# -----------------------------------
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
trained_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

print("VPN model files loaded successfully.")

# -----------------------------------
# Load input file
# -----------------------------------
df = pd.read_csv(INPUT_FILE_PATH, low_memory=False)
df.columns = df.columns.str.strip().str.lower()

target_col = "class1"
if target_col in df.columns:
    df = df.drop(columns=[target_col])

# -----------------------------------
# Handle missing values
# -----------------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# -----------------------------------
# One-hot encode
# -----------------------------------
X_input = pd.get_dummies(df)

# Add missing columns
for col in trained_feature_columns:
    if col not in X_input.columns:
        X_input[col] = 0

# Keep exact training order
X_input = X_input[trained_feature_columns]

# -----------------------------------
# Predict
# -----------------------------------
y_pred_encoded = model.predict(X_input)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

print("\nFirst 20 VPN predictions:")
print(y_pred_labels[:20])

print("\nVPN Prediction Counts:")
print(pd.Series(y_pred_labels).value_counts())