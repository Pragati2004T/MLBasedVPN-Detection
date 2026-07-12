import os
import pandas as pd

# Load cleaned UNSW dataset
df = pd.read_csv("data/processed/unsw_merged_cleaned.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Keep only IDS feature columns
ids_df = df[["proto", "service", "state", "sbytes", "dbytes"]].copy()

# Take first 100 rows
ids_sample = ids_df.head(100)

# Create uploads folder if not present
os.makedirs("data/uploads", exist_ok=True)

# Save IDS-style sample input
output_path = "data/uploads/sample_ids_input.csv"
ids_sample.to_csv(output_path, index=False)

print("IDS-style sample file created successfully.")
print(f"Saved at: {output_path}")
print("Shape:", ids_sample.shape)
print("\nColumns:")
print(ids_sample.columns.tolist())
print("\nFirst 5 rows:")
print(ids_sample.head())