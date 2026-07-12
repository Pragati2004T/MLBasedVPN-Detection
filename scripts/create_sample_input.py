import pandas as pd
import os

df = pd.read_csv("data/processed/unsw_merged_cleaned.csv")
df.columns = df.columns.str.strip().str.lower()

sample_df = df[["proto", "service", "state", "sbytes", "dbytes"]].head(100)

os.makedirs("data/uploads", exist_ok=True)
sample_df.to_csv("data/uploads/sample_ids_input.csv", index=False)

print("Sample input file created successfully.")