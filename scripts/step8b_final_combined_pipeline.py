import os
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# -----------------------------------
# Paths — resolved from project root
# -----------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IDS_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "ids_rf_model.joblib")
IDS_LABEL_ENCODER_PATH = os.path.join(_PROJECT_ROOT, "models", "ids_rf_label_encoder.joblib")
IDS_FEATURE_COLUMNS_PATH = os.path.join(_PROJECT_ROOT, "models", "ids_rf_feature_columns.joblib")
IDS_TRAIN_DIST_PATH = os.path.join(_PROJECT_ROOT, "models", "ids_rf_training_distribution.joblib")

VPN_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "vpn_rf_model.joblib")
VPN_LABEL_ENCODER_PATH = os.path.join(_PROJECT_ROOT, "models", "vpn_label_encoder.joblib")
VPN_FEATURE_COLUMNS_PATH = os.path.join(_PROJECT_ROOT, "models", "vpn_feature_columns.joblib")

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# Define schema signatures
# -----------------------------------
IDS_RAW_COLUMNS = {"proto", "service", "state", "sbytes", "dbytes"}
VPN_RAW_COLUMNS = {
    "duration",
    "total_fiat",
    "total_biat",
    "min_flowiat",
    "max_flowiat",
    "mean_flowiat",
    "std_flowiat",
    "flowbytespersecond",
}

# -----------------------------------
# Lazy model loading
# -----------------------------------
_ids_artifacts = {}
_vpn_artifacts = {}


def _load_ids_artifacts():
    """Load IDS model artifacts on first call. Returns (ok, warnings)."""
    if _ids_artifacts:
        return True, []

    paths = {
        "model": IDS_MODEL_PATH,
        "label_encoder": IDS_LABEL_ENCODER_PATH,
        "feature_columns": IDS_FEATURE_COLUMNS_PATH,
        "training_distribution": IDS_TRAIN_DIST_PATH,
    }
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        return False, [f"IDS model files missing: {', '.join(missing)}. Re-run step5b training."]

    _ids_artifacts["model"] = joblib.load(IDS_MODEL_PATH)
    _ids_artifacts["label_encoder"] = joblib.load(IDS_LABEL_ENCODER_PATH)
    _ids_artifacts["feature_columns"] = joblib.load(IDS_FEATURE_COLUMNS_PATH)
    _ids_artifacts["training_distribution"] = joblib.load(IDS_TRAIN_DIST_PATH)
    return True, []


def _load_vpn_artifacts():
    """Load VPN model artifacts on first call. Returns (ok, warnings)."""
    if _vpn_artifacts:
        return True, []

    paths = {
        "model": VPN_MODEL_PATH,
        "label_encoder": VPN_LABEL_ENCODER_PATH,
        "feature_columns": VPN_FEATURE_COLUMNS_PATH,
    }
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        return False, [f"VPN model files missing: {', '.join(missing)}. Re-run step_vpn_2 training."]

    _vpn_artifacts["model"] = joblib.load(VPN_MODEL_PATH)
    _vpn_artifacts["label_encoder"] = joblib.load(VPN_LABEL_ENCODER_PATH)
    _vpn_artifacts["feature_columns"] = joblib.load(VPN_FEATURE_COLUMNS_PATH)
    return True, []


# -----------------------------------
# Detect input schema
# -----------------------------------
def detect_input_type(df):
    cols = set(df.columns.str.strip().str.lower())

    ids_matched = IDS_RAW_COLUMNS.intersection(cols)
    vpn_matched = VPN_RAW_COLUMNS.intersection(cols)
    ids_missing = IDS_RAW_COLUMNS - cols
    vpn_missing = VPN_RAW_COLUMNS - cols

    ids_detected = len(ids_matched) >= 4
    vpn_detected = len(vpn_matched) >= 4

    if ids_detected and vpn_detected:
        input_type = "mixed"
    elif ids_detected:
        input_type = "ids"
    elif vpn_detected:
        input_type = "vpn"
    else:
        input_type = "unsupported"

    detection_detail = {
        "ids_matched": sorted(ids_matched),
        "ids_missing": sorted(ids_missing),
        "vpn_matched": sorted(vpn_matched),
        "vpn_missing": sorted(vpn_missing),
    }

    return input_type, len(ids_matched), len(vpn_matched), detection_detail


# -----------------------------------
# IDS preprocessing
# -----------------------------------
def preprocess_for_ids(df):
    temp_df = df.copy()
    temp_df.columns = temp_df.columns.str.strip().str.lower()

    missing_cols = [col for col in IDS_RAW_COLUMNS if col not in temp_df.columns]
    if missing_cols:
        raise ValueError(f"Missing IDS columns: {missing_cols}")

    X = temp_df[["proto", "service", "state", "sbytes", "dbytes"]].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("unknown").astype(str).str.strip().str.lower()
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            col_median = X[col].median()
            X[col] = X[col].fillna(col_median if pd.notna(col_median) else 0)

    X = pd.get_dummies(X)
    X = X.reindex(columns=_ids_artifacts["feature_columns"], fill_value=0)

    return X


# -----------------------------------
# VPN preprocessing
# -----------------------------------
def preprocess_for_vpn(df, drop_ids_columns=False):
    temp_df = df.copy()
    temp_df.columns = temp_df.columns.str.strip().str.lower()

    if "class1" in temp_df.columns:
        temp_df = temp_df.drop(columns=["class1"])

    # When input is mixed, drop IDS-only columns to avoid noise
    if drop_ids_columns:
        ids_only = IDS_RAW_COLUMNS - VPN_RAW_COLUMNS
        temp_df = temp_df.drop(columns=[c for c in ids_only if c in temp_df.columns])

    X = temp_df.copy()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("unknown").astype(str).str.strip().str.lower()
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            col_median = X[col].median()
            X[col] = X[col].fillna(col_median if pd.notna(col_median) else 0)

    X = pd.get_dummies(X)
    X = X.reindex(columns=_vpn_artifacts["feature_columns"], fill_value=0)

    return X


# -----------------------------------
# Concept drift detection
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
    }


# -----------------------------------
# Main combined pipeline
# -----------------------------------
def run_combined_pipeline(input_file_path):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    if input_file_path.endswith(".xlsx"):
        df = pd.read_excel(input_file_path)
    else:
        df = pd.read_csv(input_file_path, low_memory=False)

    # --- Input validation ---
    if df.empty:
        raise ValueError("Uploaded file contains no data rows.")

    # Deduplicate column names after normalization
    df.columns = df.columns.str.strip().str.lower()
    dupe_warning = None
    if df.columns.duplicated().any():
        dupes = list(df.columns[df.columns.duplicated(keep=False)].unique())
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        dupe_warning = f"Duplicate columns detected after normalization: {dupes}. Kept first occurrence of each."

    original_df = df.copy()

    input_type, ids_match_count, vpn_match_count, detection_detail = detect_input_type(df)

    results = {
        "total_records": len(df),
        "input_type": input_type,
        "attack_counts": {},
        "vpn_counts": {},
        "concept_drift": {},
        "warnings": [],
        "output_file": "",
        "detection_detail": detection_detail,
        "data_preview": df.head(10).to_dict("records"),
        "data_columns": list(df.columns),
    }

    if dupe_warning:
        results["warnings"].append(dupe_warning)

    # -----------------------------------
    # IDS-style file
    # -----------------------------------
    if input_type == "ids":
        ok, load_warnings = _load_ids_artifacts()
        results["warnings"].extend(load_warnings)

        if ok:
            try:
                X_ids = preprocess_for_ids(df)
                y_ids_pred_encoded = _ids_artifacts["model"].predict(X_ids)
                ids_predictions = _ids_artifacts["label_encoder"].inverse_transform(y_ids_pred_encoded)

                results["attack_counts"] = pd.Series(ids_predictions).value_counts().to_dict()
                results["concept_drift"] = detect_concept_drift(
                    _ids_artifacts["training_distribution"], ids_predictions
                )

                original_df["predicted_attack"] = ids_predictions

            except Exception as e:
                results["warnings"].append(f"IDS prediction failed: {str(e)}")

        results["warnings"].append(
            f"VPN module skipped — file matches IDS schema "
            f"({ids_match_count}/{len(IDS_RAW_COLUMNS)} IDS columns found)."
        )

        if detection_detail["ids_missing"]:
            results["warnings"].append(
                f"IDS columns not found in upload: {detection_detail['ids_missing']}. "
                f"These were filled with defaults."
            )

    # -----------------------------------
    # VPN-style file
    # -----------------------------------
    elif input_type == "vpn":
        ok, load_warnings = _load_vpn_artifacts()
        results["warnings"].extend(load_warnings)

        if ok:
            try:
                X_vpn = preprocess_for_vpn(df)
                y_vpn_pred_encoded = _vpn_artifacts["model"].predict(X_vpn)
                vpn_predictions = _vpn_artifacts["label_encoder"].inverse_transform(y_vpn_pred_encoded)

                results["vpn_counts"] = pd.Series(vpn_predictions).value_counts().to_dict()
                original_df["predicted_vpn"] = vpn_predictions

            except Exception as e:
                results["warnings"].append(f"VPN prediction failed: {str(e)}")

        results["warnings"].append(
            f"IDS module skipped — file matches VPN schema "
            f"({vpn_match_count}/{len(VPN_RAW_COLUMNS)} VPN columns found). "
            f"Concept drift detection is only available for IDS-compatible files."
        )

    # -----------------------------------
    # Mixed schema file
    # -----------------------------------
    elif input_type == "mixed":
        results["warnings"].append(
            "Mixed schema detected — file contains both IDS and VPN columns. "
            "Both models will run independently, but results may be less reliable "
            "unless both feature sets come from the same real network capture."
        )

        # Run IDS
        ids_ok, ids_load_warnings = _load_ids_artifacts()
        results["warnings"].extend(ids_load_warnings)

        if ids_ok:
            try:
                X_ids = preprocess_for_ids(df)
                y_ids_pred_encoded = _ids_artifacts["model"].predict(X_ids)
                ids_predictions = _ids_artifacts["label_encoder"].inverse_transform(y_ids_pred_encoded)

                results["attack_counts"] = pd.Series(ids_predictions).value_counts().to_dict()
                results["concept_drift"] = detect_concept_drift(
                    _ids_artifacts["training_distribution"], ids_predictions
                )

                original_df["predicted_attack"] = ids_predictions

            except Exception as e:
                results["warnings"].append(f"IDS prediction failed on mixed input: {str(e)}")

        # Run VPN — drop IDS-only columns to avoid encoding noise
        vpn_ok, vpn_load_warnings = _load_vpn_artifacts()
        results["warnings"].extend(vpn_load_warnings)

        if vpn_ok:
            try:
                X_vpn = preprocess_for_vpn(df, drop_ids_columns=True)
                y_vpn_pred_encoded = _vpn_artifacts["model"].predict(X_vpn)
                vpn_predictions = _vpn_artifacts["label_encoder"].inverse_transform(y_vpn_pred_encoded)

                results["vpn_counts"] = pd.Series(vpn_predictions).value_counts().to_dict()
                original_df["predicted_vpn"] = vpn_predictions

            except Exception as e:
                results["warnings"].append(f"VPN prediction failed on mixed input: {str(e)}")

    # -----------------------------------
    # Unsupported file
    # -----------------------------------
    else:
        results["warnings"].append(
            f"Unsupported dataset schema. "
            f"IDS columns matched: {ids_match_count}/{len(IDS_RAW_COLUMNS)} {detection_detail['ids_matched']}. "
            f"VPN columns matched: {vpn_match_count}/{len(VPN_RAW_COLUMNS)} {detection_detail['vpn_matched']}. "
            f"At least 4 columns from either schema are required."
        )

    # -----------------------------------
    # Save output — only if predictions were produced
    # -----------------------------------
    has_predictions = "predicted_attack" in original_df.columns or "predicted_vpn" in original_df.columns
    if has_predictions:
        output_file = os.path.join(OUTPUT_DIR, "final_combined_predictions.csv")
        original_df.to_csv(output_file, index=False)
        results["output_file"] = output_file

    return results
