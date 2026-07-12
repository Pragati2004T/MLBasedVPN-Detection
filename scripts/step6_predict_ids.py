import os
import joblib
import pandas as pd

# -----------------------------------
# Paths
# -----------------------------------
MODEL_PATH = "models/ids_rf_model.joblib"
LABEL_ENCODER_PATH = "models/ids_rf_label_encoder.joblib"
FEATURE_COLUMNS_PATH = "models/ids_rf_feature_columns.joblib"

# Test upload file path
INPUT_FILE_PATH = "data/uploads/sample_ids_input.csv"

# -----------------------------------
# Load saved model and helper files
# -----------------------------------
print("Loading model and helper files...")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
trained_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

print("Model loaded successfully.")
print("Number of trained feature columns:", len(trained_feature_columns))

# -----------------------------------
# Load input CSV file
# -----------------------------------
if not os.path.exists(INPUT_FILE_PATH):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE_PATH}")

print("\nLoading input file...")
df = pd.read_csv(INPUT_FILE_PATH)

print("Input file loaded successfully.")
print("Input shape:", df.shape)

# -----------------------------------
# Normalize column names
# -----------------------------------
df.columns = df.columns.str.strip().str.lower()

print("\nInput columns:")
print(df.columns.tolist())

# -----------------------------------
# Keep only required raw columns
# These are the original columns used before get_dummies
# -----------------------------------
required_raw_columns = ["proto", "service", "state", "sbytes", "dbytes"]

missing_raw_columns = [col for col in required_raw_columns if col not in df.columns]

if missing_raw_columns:
    raise ValueError(f"Missing required columns in input file: {missing_raw_columns}")

X_input = df[required_raw_columns].copy()

print("\nSelected raw input columns:")
print(X_input.columns.tolist())

# -----------------------------------
# Handle missing values in raw input
# -----------------------------------
for col in X_input.columns:
    if X_input[col].dtype == "object":
        X_input[col] = X_input[col].fillna("unknown").astype(str).str.strip().str.lower()
    else:
        X_input[col] = pd.to_numeric(X_input[col], errors="coerce")
        X_input[col] = X_input[col].fillna(X_input[col].median())

# -----------------------------------
# Apply one-hot encoding
# -----------------------------------
X_input_encoded = pd.get_dummies(X_input)

print("\nShape after one-hot encoding:", X_input_encoded.shape)

# -----------------------------------
# Add missing columns that existed during training
# -----------------------------------
for col in trained_feature_columns:
    if col not in X_input_encoded.columns:
        X_input_encoded[col] = 0

# -----------------------------------
# Remove extra columns not used in training
# -----------------------------------
X_input_encoded = X_input_encoded[trained_feature_columns]

print("Shape after aligning with training columns:", X_input_encoded.shape)

# -----------------------------------
# Run prediction
# -----------------------------------
print("\nRunning IDS prediction...")
y_pred_encoded = model.predict(X_input_encoded)

# Convert numeric predictions back to attack names
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# -----------------------------------
# Create output dataframe
# -----------------------------------
output_df = df.copy()
output_df["predicted_attack"] = y_pred_labels

# -----------------------------------
# Show results
# -----------------------------------
print("\nPrediction completed.")
print("\nFirst 10 predictions:")
print(output_df[["predicted_attack"]].head(10))

attack_counts = output_df["predicted_attack"].value_counts()

print("\nTotal Records Processed:", len(output_df))
print("\nAttack Type Counts:")
print(attack_counts)

# -----------------------------------
# Save prediction results
# -----------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_file_path = os.path.join(OUTPUT_DIR, "ids_predictions_output.csv")
output_df.to_csv(output_file_path, index=False)

print(f"\nPrediction results saved to: {output_file_path}")