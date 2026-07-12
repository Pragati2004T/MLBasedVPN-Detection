# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VPN-Aware Intrusion Detection System — a dual-model ML pipeline that classifies network traffic using:
1. **IDS model** (Random Forest, 200 trees): classifies attack types from UNSW-NB15 dataset features (proto, service, state, sbytes, dbytes)
2. **VPN model** (Random Forest, 200 trees): classifies VPN vs non-VPN traffic from flow-level features (duration, flow IAT metrics, throughput)

Both models use `joblib` for serialization and `pd.get_dummies` for one-hot encoding. Concept drift is monitored via Jensen-Shannon distance (threshold: 0.20) comparing training vs prediction class distributions.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Data preprocessing
python scripts/step4_merge_clean_unsw.py      # Merge/clean 4 UNSW-NB15 CSVs → data/processed/unsw_merged_cleaned.csv
python scripts/step_vpn_1_merge_csv_files.py   # Merge VPN scenario CSVs → data/processed/vpn_merged.csv

# Train models
python scripts/step5b_train_rf_ids_model.py    # Train IDS Random Forest (production model)
python scripts/step_vpn_2_train_model.py       # Train VPN Random Forest

# Alternative IDS models (not used in production pipeline)
python step5_train_ids_model.py                # MLP model
python step5c_train_xgboost_ids_model.py       # XGBoost model

# Run predictions standalone
python scripts/step6_predict_ids.py            # IDS-only prediction
python scripts/step_vpn_3_predict.py           # VPN-only prediction
python scripts/step7_concept_drift.py          # Standalone drift detection

# Generate test data
python scripts/create_ids_test_file.py         # Sample IDS input → data/uploads/sample_ids_input.csv

# Run the Streamlit web app
streamlit run app/app.py
```

## Architecture

### Pipeline Flow

Raw data → preprocessing scripts (step4, vpn_1) → processed CSVs → training scripts (step5b, vpn_2) → saved models in `models/` → inference pipelines (step8/step8b) → predictions in `outputs/`

### Key Entry Points

- **`app/app.py`** — Streamlit dashboard. Accepts CSV/Excel uploads, auto-detects input type, runs the combined pipeline, displays SOC analytics (risk score, threat level, attack distribution, VPN distribution, concept drift status), and offers CSV download.
- **`scripts/step8b_final_combined_pipeline.py`** — Production pipeline called by the app. `run_combined_pipeline(input_file_path)` auto-detects whether input is IDS, VPN, mixed, or unsupported based on column overlap, runs appropriate model(s), and returns a results dict with predictions, drift metrics, and warnings.
- **`scripts/step8_final_pipeline.py`** — IDS-only pipeline with drift detection. `run_ids_pipeline(input_file_path)`.

### Input Type Detection (step8b)

The combined pipeline checks column overlap against known feature sets:
- IDS: ≥4 of {proto, service, state, sbytes, dbytes}
- VPN: ≥4 of {duration, total_fiat, total_biat, min_flowiat, max_flowiat, mean_flowiat, std_flowiat, flowbytespersecond}

### Model Artifacts (all in `models/`)

| Artifact | Purpose |
|----------|---------|
| `ids_rf_model.joblib` | IDS Random Forest model |
| `ids_rf_label_encoder.joblib` | IDS attack category encoder |
| `ids_rf_feature_columns.joblib` | IDS one-hot encoded column list |
| `ids_rf_training_distribution.joblib` | IDS class distribution for drift detection |
| `vpn_rf_model.joblib` | VPN Random Forest model |
| `vpn_label_encoder.joblib` | VPN class encoder |
| `vpn_feature_columns.joblib` | VPN one-hot encoded column list |

### Prediction Alignment Pattern

All prediction scripts follow the same pattern to handle feature mismatch between input data and trained model:
1. One-hot encode input with `pd.get_dummies`
2. Add any missing training columns (set to 0)
3. Reindex to exact training column order via saved feature columns

### Risk Score Formula (app.py)

```
risk_score = min(100, attack_pct * 0.5 + vpn_pct * 0.2 + js_distance * 100 * 0.3)
Threat: Low (<35), Medium (35-70), High (≥70); escalated to Medium if drift detected
```

## Data Layout

- `data/raw/unsw/` — UNSW-NB15_1.csv through UNSW-NB15_4.csv
- `data/raw/vpn/` — Scenario A1, A2, B subdirectories with CSVs
- `data/processed/` — Cleaned/merged datasets
- `data/uploads/` — Sample/test input files
- `outputs/` — Prediction CSVs and drift results
