import os
import pandas as pd

# -----------------------------------
# Paths
# -----------------------------------
VPN_BASE_DIR = "data/raw/vpn"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------------------
# Folders containing VPN CSV files
# -----------------------------------
VPN_FOLDERS = [
    os.path.join(VPN_BASE_DIR, "Scenario A1"),
    os.path.join(VPN_BASE_DIR, "Scenarios A2"),
    os.path.join(VPN_BASE_DIR, "Scenario B"),
]

all_dfs = []
total_files_loaded = 0

# -----------------------------------
# Read all CSV files from all folders
# -----------------------------------
for folder in VPN_FOLDERS:
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    print(f"\nChecking folder: {folder}")

    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in this folder.")
        continue

    for file_name in csv_files:
        file_path = os.path.join(folder, file_name)

        try:
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.strip().str.lower()

            print(f"Loaded: {file_name} | Shape: {df.shape}")

            all_dfs.append(df)
            total_files_loaded += 1

        except Exception as e:
            print(f"Error loading {file_name}: {e}")

# -----------------------------------
# Merge all files
# -----------------------------------
if not all_dfs:
    raise ValueError("No VPN CSV files were loaded. Check your folder paths and file names.")

merged_df = pd.concat(all_dfs, ignore_index=True)

print("\nAll VPN CSV files merged successfully.")
print("Total files loaded:", total_files_loaded)
print("Merged shape before cleaning:", merged_df.shape)

print("\nMerged columns:")
print(merged_df.columns.tolist())

# -----------------------------------
# Remove duplicate rows
# -----------------------------------
merged_df.drop_duplicates(inplace=True)

print("\nShape after removing duplicates:", merged_df.shape)

# -----------------------------------
# Save merged file
# -----------------------------------
output_path = os.path.join(PROCESSED_DIR, "vpn_merged.csv")
merged_df.to_csv(output_path, index=False)

print(f"\nMerged VPN dataset saved to: {output_path}")
print("\nFirst 5 rows:")
print(merged_df.head())