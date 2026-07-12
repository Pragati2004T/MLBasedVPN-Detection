import os
import pandas as pd

# -----------------------------------
# Paths
# -----------------------------------
RAW_UNSW_DIR = "data/raw/unsw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

UNSW_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv"
]

all_dfs = []

# -----------------------------------
# Read all UNSW files
# -----------------------------------
for file_name in UNSW_FILES:
    file_path = os.path.join(RAW_UNSW_DIR, file_name)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    print(f"\nReading file: {file_name}")

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception:
        df = pd.read_csv(file_path, encoding="latin1", low_memory=False)

    print(f"Original shape of {file_name}: {df.shape}")
    all_dfs.append(df)

if not all_dfs:
    raise ValueError("No UNSW files were loaded. Check the raw dataset folder.")

# -----------------------------------
# Merge all files
# -----------------------------------
merged_df = pd.concat(all_dfs, ignore_index=True)
print("\nMerged shape before cleaning:", merged_df.shape)

# Normalize column names
merged_df.columns = merged_df.columns.str.strip().str.lower()
print("\nNormalized columns:")
print(merged_df.columns.tolist())

# Remove duplicates
merged_df.drop_duplicates(inplace=True)
print("\nShape after removing duplicates:", merged_df.shape)

# -----------------------------------
# Required columns
# -----------------------------------
required_cols = ["proto", "service", "state", "sbytes", "dbytes", "attack_cat", "label"]
missing_required = [col for col in required_cols if col not in merged_df.columns]

if missing_required:
    raise ValueError(f"Missing required columns in UNSW data: {missing_required}")

clean_df = merged_df[required_cols].copy()
print("\nShape after selecting required columns:", clean_df.shape)

# -----------------------------------
# Basic feature cleaning
# -----------------------------------
# Clean categorical columns
for col in ["proto", "service", "state"]:
    clean_df[col] = clean_df[col].astype(str).str.strip()
    clean_df[col] = clean_df[col].replace(["", "nan", "None", "NULL"], pd.NA)
    clean_df[col] = clean_df[col].fillna("unknown_feature")

# Clean numeric columns
for col in ["sbytes", "dbytes", "label"]:
    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

clean_df["sbytes"] = clean_df["sbytes"].fillna(clean_df["sbytes"].median())
clean_df["dbytes"] = clean_df["dbytes"].fillna(clean_df["dbytes"].median())

# Drop rows where label is missing because label is essential
clean_df = clean_df.dropna(subset=["label"])
clean_df["label"] = clean_df["label"].astype(int)

# -----------------------------------
# Clean attack_cat carefully
# -----------------------------------
clean_df["attack_cat"] = clean_df["attack_cat"].astype(str).str.strip()

# Convert obvious empty-like values to NA
clean_df["attack_cat"] = clean_df["attack_cat"].replace(
    ["", " ", "nan", "None", "NULL", "-", "--"],
    pd.NA
)

# Standardize label names
clean_df["attack_cat"] = clean_df["attack_cat"].replace({
    "Backdoors": "Backdoor",
    "Reconnaissance ": "Reconnaissance",
    " Reconnaissance": "Reconnaissance",
    "Normal ": "Normal"
})

# If attack_cat is missing and label == 0, this is most likely normal traffic
clean_df.loc[
    clean_df["attack_cat"].isna() & (clean_df["label"] == 0),
    "attack_cat"
] = "Normal"

# If attack_cat is missing and label == 1, drop those rows because label is attack but class is unknown
before_drop = len(clean_df)
clean_df = clean_df.dropna(subset=["attack_cat"])
after_drop = len(clean_df)

print(f"\nDropped rows with unresolved missing attack_cat: {before_drop - after_drop}")

# Final strip
clean_df["attack_cat"] = clean_df["attack_cat"].astype(str).str.strip()

# -----------------------------------
# Optional: remove impossible labels
# -----------------------------------
valid_attack_labels = {
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms"
}

before_filter = len(clean_df)
clean_df = clean_df[clean_df["attack_cat"].isin(valid_attack_labels)].copy()
after_filter = len(clean_df)

print(f"Removed rows with invalid attack_cat labels: {before_filter - after_filter}")

# -----------------------------------
# Final checks
# -----------------------------------
print("\nFinal shape:", clean_df.shape)
print("\nattack_cat value counts:")
print(clean_df["attack_cat"].value_counts())

print("\nlabel value counts:")
print(clean_df["label"].value_counts())

# -----------------------------------
# Save cleaned dataset
# -----------------------------------
output_path = os.path.join(PROCESSED_DIR, "unsw_merged_cleaned.csv")
clean_df.to_csv(output_path, index=False)

print(f"\nCleaned UNSW dataset saved to: {output_path}")
print("\nFirst 5 rows:")
print(clean_df.head())