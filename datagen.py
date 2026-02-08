"""
Sample data generation and import script for creating a Delta Lake table in LakeFS.

This script:
- Generates a synthetic tabular classification dataset (10,000 rows):
  - 20 numerical features (informative + noise)
  - 1 categorical feature ("color" with values 'red', 'blue', 'green')
  - Target column "label" (integer classes 0, 1, 2)
- Writes the data as a Delta Lake table to the same LakeFS path used in the training script.
- Uses overwrite mode to create/replace the table (creates a new version in LakeFS).

Installation (run once):
pip install deltalake pandas scikit-learn

Run this script once to populate the table, then run the training script.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from deltalake.writer import write_deltalake

# ========================= CONFIGURATION =========================
# Use the SAME LakeFS settings as your training script
LAKEFS_ENDPOINT = "http://your-lakefs-host:8000"          # e.g. http://localhost:8000
LAKEFS_ACCESS_KEY = "your-lakefs-access-key"
LAKEFS_SECRET_KEY = "your-lakefs-secret-key"

REPO = "example-repo"
BRANCH = "main"
TABLE_PATH = "data/table"

DELTA_URI = f"s3://{REPO}/{BRANCH}/{TABLE_PATH}"

STORAGE_OPTIONS = {
    "AWS_ENDPOINT_URL": LAKEFS_ENDPOINT,
    "AWS_ACCESS_KEY_ID": LAKEFS_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": LAKEFS_SECRET_KEY,
    "AWS_REGION": "us-east-1",
    "AWS_S3_FORCE_PATH_STYLE": "true",
}

# Dataset size
N_SAMPLES = 10000
RANDOM_STATE = 42

# ========================= GENERATE SAMPLE DATA =========================
print("Generating synthetic dataset...")

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=2,
    random_state=RANDOM_STATE,
)

# Create DataFrame with meaningful column names
columns = [f"num_feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=columns)

# Add one categorical feature
np.random.seed(RANDOM_STATE)
df["color"] = np.random.choice(["red", "blue", "green"], size=N_SAMPLES)

# Add target column
df["label"] = y

print(f"Generated DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())
print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

# ========================= WRITE TO DELTA LAKE =========================
print(f"\nWriting to Delta Lake table: {DELTA_URI}")

try:
    write_deltalake(
        DELTA_URI,
        df,
        mode="overwrite",                  # Replace existing table (creates new version)
        storage_options=STORAGE_OPTIONS,
        schema={"label": "int64"},         # Optional: enforce types if needed
    )
    print("✅ Successfully wrote Delta table!")
    print("   A new version has been created in LakeFS.")
except Exception as e:
    print(f"❌ Failed to write Delta table: {e}")
    print("   Check credentials, endpoint, and write permissions on the branch.")

# ========================= VERIFICATION =========================
# Optional: read back a small sample to verify
try:
    from deltalake import DeltaTable
    dt = DeltaTable(DELTA_URI, storage_options=STORAGE_OPTIONS)
    print(f"\nVerification: Table version {dt.version()}")
    sample = dt.to_pyarrow_dataset().head(5).to_pandas()
    print("Sample from written table:")
    print(sample)
except Exception as e:
    print(f"\nCould not verify (optional): {e}")