import os
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# -----------------------------------
# Paths
# -----------------------------------
TRAIN_DIST_PATH = "models/ids_rf_training_distribution.joblib"
PREDICTION_FILE_PATH = "outputs/ids_predictions_output.csv"

# -----------------------------------
# Load training distribution
# -----------------------------------
if not os.path.exists(TRAIN_DIST_PATH):
    raise FileNotFoundError(f"Training distribution file not found: {TRAIN_DIST_PATH}")

training_distribution = joblib.load(TRAIN_DIST_PATH)

print("Training distribution loaded successfully.")
print("\nTraining distribution:")
print(training_distribution)

# -----------------------------------
# Load prediction output
# -----------------------------------
if not os.path.exists(PREDICTION_FILE_PATH):
    raise FileNotFoundError(f"Prediction output file not found: {PREDICTION_FILE_PATH}")

pred_df = pd.read_csv(PREDICTION_FILE_PATH)

if "predicted_attack" not in pred_df.columns:
    raise ValueError("Column 'predicted_attack' not found in prediction output file.")

print("\nPrediction output loaded successfully.")
print("Prediction output shape:", pred_df.shape)

# -----------------------------------
# Build current prediction distribution
# -----------------------------------
prediction_distribution = pred_df["predicted_attack"].value_counts(normalize=True).to_dict()

print("\nPrediction distribution:")
print(prediction_distribution)

# -----------------------------------
# Align both distributions
# -----------------------------------
all_classes = sorted(set(training_distribution.keys()) | set(prediction_distribution.keys()))

train_probs = np.array([training_distribution.get(cls, 0.0) for cls in all_classes], dtype=float)
pred_probs = np.array([prediction_distribution.get(cls, 0.0) for cls in all_classes], dtype=float)

# Normalize safely
if train_probs.sum() > 0:
    train_probs = train_probs / train_probs.sum()

if pred_probs.sum() > 0:
    pred_probs = pred_probs / pred_probs.sum()

print("\nAligned classes:")
print(all_classes)

print("\nTraining probability vector:")
print(train_probs)

print("\nPrediction probability vector:")
print(pred_probs)

# -----------------------------------
# Calculate Jensen-Shannon Distance
# -----------------------------------
js_distance = jensenshannon(train_probs, pred_probs)

print("\nJensen-Shannon Distance:", js_distance)

# -----------------------------------
# Decide drift
# -----------------------------------
THRESHOLD = 0.20

if js_distance > THRESHOLD:
    drift_status = "Concept Drift Detected"
else:
    drift_status = "No Concept Drift Detected"

print("\nDrift Decision:")
print(drift_status)
print(f"Threshold used: {THRESHOLD}")

# -----------------------------------
# Save drift result to file
# -----------------------------------
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

drift_result = pd.DataFrame([{
    "js_distance": float(js_distance),
    "threshold": THRESHOLD,
    "drift_status": drift_status
}])

drift_output_path = os.path.join(output_dir, "concept_drift_result.csv")
drift_result.to_csv(drift_output_path, index=False)

print(f"\nConcept drift result saved to: {drift_output_path}")