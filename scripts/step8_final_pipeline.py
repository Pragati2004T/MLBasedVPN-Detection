import os
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# -----------------------------------
# Paths
# -----------------------------------
MODEL_PATH = "models/ids_rf_model.joblib"
LABEL_ENCODER_PATH = "models/ids_rf_label_encoder.joblib"
FEATURE_COLUMNS_PATH = "models/ids_rf_feature_columns.joblib"
TRAIN_DIST_PATH = "models/ids_rf_training_distribution.joblib"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# Load model and helper files once
# -----------------------------------
print("Loading model and helper files...")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
trained_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
training_distribution = joblib.load(TRAIN_DIST_PATH)

print("All model files loaded successfully.")

# -----------------------------------
# Helper function: preprocess input
# -----------------------------------
def preprocess_input_file(df):
    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    required_raw_columns = ["proto", "service", "state", "sbytes", "dbytes"]

    missing_raw_columns = [col for col in required_raw_columns if col not in df.columns]
    if missing_raw_columns:
        raise ValueError(f"Missing required columns: {missing_raw_columns}")

    X_input = df[required_raw_columns].copy()

    # Handle missing values
    for col in X_input.columns:
        if X_input[col].dtype == "object":
            X_input[col] = X_input[col].fillna("unknown").astype(str).str.strip().str.lower()
        else:
            X_input[col] = pd.to_numeric(X_input[col], errors="coerce")
            X_input[col] = X_input[col].fillna(X_input[col].median())

    # One-hot encoding
    X_input_encoded = pd.get_dummies(X_input)

    # Add missing columns from training
    for col in trained_feature_columns:
        if col not in X_input_encoded.columns:
            X_input_encoded[col] = 0

    # Keep only training columns in exact order
    X_input_encoded = X_input_encoded[trained_feature_columns]

    return X_input_encoded, df

# -----------------------------------
# Helper function: detect drift
# -----------------------------------
def detect_concept_drift(training_dist, prediction_labels, threshold=0.20):
    prediction_distribution = pd.Series(prediction_labels).value_counts(normalize=True).to_dict()

    all_classes = sorted(set(training_dist.keys()) | set(prediction_distribution.keys()))

    train_probs = np.array([training_dist.get(cls, 0.0) for cls in all_classes], dtype=float)
    pred_probs = np.array([prediction_distribution.get(cls, 0.0) for cls in all_classes], dtype=float)

    if train_probs.sum() > 0:
        train_probs = train_probs / train_probs.sum()

    if pred_probs.sum() > 0:
        pred_probs = pred_probs / pred_probs.sum()

    js_distance = jensenshannon(train_probs, pred_probs)

    drift_detected = js_distance > threshold

    return {
        "js_distance": float(js_distance),
        "threshold": threshold,
        "drift_detected": drift_detected,
        "drift_status": "Concept Drift Detected" if drift_detected else "No Concept Drift Detected",
        "prediction_distribution": prediction_distribution
    }

# -----------------------------------
# Main pipeline function
# -----------------------------------
def run_ids_pipeline(input_file_path):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    print(f"\nLoading input file: {input_file_path}")
    df = pd.read_csv(input_file_path)

    print("Input file loaded successfully.")
    print("Input shape:", df.shape)

    # Preprocess
    X_processed, original_df = preprocess_input_file(df)

    print("Preprocessing completed.")
    print("Processed shape:", X_processed.shape)

    # Predict
    print("Running prediction...")
    y_pred_encoded = model.predict(X_processed)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    # Create output dataframe
    output_df = original_df.copy()
    output_df["predicted_attack"] = y_pred_labels

    # Attack counts
    attack_counts = output_df["predicted_attack"].value_counts().to_dict()

    # Total records
    total_records = len(output_df)

    # Drift detection
    drift_result = detect_concept_drift(training_distribution, y_pred_labels)

    # Save prediction file
    output_prediction_path = os.path.join(OUTPUT_DIR, "final_ids_predictions.csv")
    output_df.to_csv(output_prediction_path, index=False)

    # Save drift summary file
    drift_summary_df = pd.DataFrame([{
        "js_distance": drift_result["js_distance"],
        "threshold": drift_result["threshold"],
        "drift_status": drift_result["drift_status"]
    }])

    drift_output_path = os.path.join(OUTPUT_DIR, "final_concept_drift_result.csv")
    drift_summary_df.to_csv(drift_output_path, index=False)

    # Final result dictionary
    result = {
        "total_records": total_records,
        "attack_counts": attack_counts,
        "concept_drift": {
            "js_distance": drift_result["js_distance"],
            "threshold": drift_result["threshold"],
            "drift_status": drift_result["drift_status"]
        },
        "prediction_output_file": output_prediction_path,
        "drift_output_file": drift_output_path
    }

    return result

# -----------------------------------
# Run directly for testing
# -----------------------------------
if __name__ == "__main__":
    test_file = "data/uploads/sample_ids_input.csv"

    results = run_ids_pipeline(test_file)

    print("\n========== FINAL RESULT ==========")
    print("Total Records Processed:", results["total_records"])

    print("\nAttack Counts:")
    for attack, count in results["attack_counts"].items():
        print(f"{attack}: {count}")

    print("\nConcept Drift Result:")
    print("JS Distance:", results["concept_drift"]["js_distance"])
    print("Threshold:", results["concept_drift"]["threshold"])
    print("Status:", results["concept_drift"]["drift_status"])

    print("\nSaved Files:")
    print(results["prediction_output_file"])
    print(results["drift_output_file"])