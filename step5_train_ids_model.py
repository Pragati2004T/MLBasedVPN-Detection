import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# Paths
# -----------------------------------
DATA_PATH = "data/processed/unsw_merged_cleaned.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------
# Load cleaned dataset
# -----------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# -----------------------------------
# Normalize column names
# -----------------------------------
df.columns = df.columns.str.strip().str.lower()

# -----------------------------------
# Check target column
# -----------------------------------
if "attack_cat" not in df.columns:
    raise ValueError("Target column 'attack_cat' not found in dataset.")

# -----------------------------------
# Define features and target
# -----------------------------------
X = df.drop(columns=["attack_cat"])

# If label exists, remove it from input features
# because attack_cat is our target for multiclass classification
if "label" in X.columns:
    X = X.drop(columns=["label"])

y = df["attack_cat"].astype(str).str.strip()

print("\nInput feature columns:")
print(X.columns.tolist())

print("\nUnique attack categories:")
print(sorted(y.unique()))

# -----------------------------------
# Encode target labels
# -----------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded classes:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i} -> {class_name}")

# -----------------------------------
# Identify numeric and categorical columns
# -----------------------------------
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric columns:")
print(numeric_cols)

print("\nCategorical columns:")
print(categorical_cols)

# -----------------------------------
# Preprocessing for numeric columns
# -----------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# -----------------------------------
# Preprocessing for categorical columns
# -----------------------------------
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# -----------------------------------
# Combine preprocessing
# -----------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# -----------------------------------
# Build full pipeline
# -----------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=50,
        random_state=42
    ))
])

# -----------------------------------
# Split data into train and test
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# -----------------------------------
# Train model
# -----------------------------------
print("\nTraining IDS model...")
model.fit(X_train, y_train)

print("Training completed.")

# -----------------------------------
# Predict on test data
# -----------------------------------
y_pred = model.predict(X_test)

# -----------------------------------
# Evaluate model
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# -----------------------------------
# Save trained model and helpers
# -----------------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "ids_mlp_model.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "ids_label_encoder.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "ids_feature_columns.joblib"))

# Save training distribution for concept drift
train_dist = pd.Series(
    label_encoder.inverse_transform(y_train)
).value_counts(normalize=True).to_dict()

joblib.dump(train_dist, os.path.join(MODEL_DIR, "ids_training_distribution.joblib"))

print("\nModel and helper files saved successfully.")
print("Saved files:")
print("- models/ids_mlp_model.joblib")
print("- models/ids_label_encoder.joblib")
print("- models/ids_feature_columns.joblib")
print("- models/ids_training_distribution.joblib")