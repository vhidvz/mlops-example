"""
Sample data generation and import script for creating a Delta Lake table in LakeFS.

This script:
- Generates a synthetic tabular classification dataset (10,000 rows):
  - 20 numerical features (informative + noise)
  - 1 categorical feature ("color" with values 'red', 'blue', 'green')
  - Target column "label" (integer classes 0, 1, 2)
- Writes the data as a Delta Lake table to the same LakeFS path used in the training script.
- Uses overwrite mode to create/replace the table (creates a new version in LakeFS).

Run this script once to populate the table, then run the training script.
"""
import os

import numpy as np
import polars as pl

from dotenv import load_dotenv
from deltalake import DeltaTable
from sklearn.datasets import make_classification


# ========================= CONFIGURATION =========================

def lakefs_config():
  LAKEFS_STORAGE_REPO = os.getenv('LAKEFS_STORAGE_REPO')
  LAKEFS_STORAGE_TABLE = os.getenv('LAKEFS_STORAGE_TABLE')
  LAKEFS_STORAGE_BRANCH = os.getenv('LAKEFS_STORAGE_BRANCH')
  
  assert LAKEFS_STORAGE_REPO, 'lakefs repo is required'
  assert LAKEFS_STORAGE_TABLE, 'lakefs table is required'
  assert LAKEFS_STORAGE_BRANCH, 'lakefs branch is required'

  LAKEFS_SERVER_ENDPOINT_URL = os.getenv('LAKEFS_SERVER_ENDPOINT_URL')
  LAKEFS_CREDENTIALS_ACCESS_KEY_ID = os.getenv('LAKEFS_CREDENTIALS_ACCESS_KEY_ID')
  LAKEFS_CREDENTIALS_SECRET_ACCESS_KEY = os.getenv('LAKEFS_CREDENTIALS_SECRET_ACCESS_KEY')
  
  assert LAKEFS_SERVER_ENDPOINT_URL, 'lakefs server endpoint url is required'
  assert LAKEFS_CREDENTIALS_ACCESS_KEY_ID, 'lakefs credential access key id is required'
  assert LAKEFS_CREDENTIALS_SECRET_ACCESS_KEY, 'lakefs credential secret access key is required'
  
  STORAGE_OPTIONS = {
    "allow_http": "true",
    "endpoint": LAKEFS_SERVER_ENDPOINT_URL,
    "access_key_id": LAKEFS_CREDENTIALS_ACCESS_KEY_ID,
    "secret_access_key": LAKEFS_CREDENTIALS_SECRET_ACCESS_KEY,
  }

  DELTA_URI = f"s3://{LAKEFS_STORAGE_REPO}/{LAKEFS_STORAGE_BRANCH}/{LAKEFS_STORAGE_TABLE}"

  return STORAGE_OPTIONS, DELTA_URI

# ========================= GENERATE SAMPLE DATA =========================

def sample_generation(n_samples = 10_000) -> pl.DataFrame:
    print("Generating synthetic dataset...")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
    )

    # Create DataFrame with meaningful column names
    columns = [f"num_feature_{i}" for i in range(X.shape[1])]
    df = pl.DataFrame(X, schema=columns)
    
    # Add categorical feature and target using hstack (cleanest way)
    df = df.hstack(
        pl.DataFrame(
            {
                "color": np.random.choice(["red", "blue", "green"], size=n_samples),
                "label": y,
            }
        )
    )
    
    print(f"Generated DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort("label"))
    
    return df

# ========================= WRITE TO DELTA LAKE =========================

def write_deltalake(df: pl.DataFrame, storage_options: dict, delta_uri: str):
    print(f"\nWriting to Delta Lake table: {delta_uri}")
    try:
        df.write_delta(
            delta_uri,
            mode="overwrite",
            storage_options=storage_options
        )
        print("Successfully wrote Delta table!")
        print("A new version has been created in LakeFS.")
    except Exception as e:
        print(f"Failed to write Delta table: {e}")

# ========================= VERIFICATION =========================

def verification(storage_options: dict, delta_uri: str):
    try:
        dt = DeltaTable(delta_uri, storage_options=storage_options)
        print(f"\nVerification: Table version {dt.version()}")
        sample = dt.to_pyarrow_dataset().head(5).to_pandas()
        print("Sample from written table:")
        print(sample)
    except Exception as e:
        print(f"\nCould not verify (optional): {e}")


if __name__ == '__main__':
    load_dotenv()
    df = sample_generation()
    STORAGE_OPTIONS, DELTA_URI = lakefs_config()
    write_deltalake(df, STORAGE_OPTIONS, DELTA_URI)
    verification(STORAGE_OPTIONS, DELTA_URI)
