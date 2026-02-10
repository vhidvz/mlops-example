import os
import sys
import json
import torch
import joblib

import numpy as np
import pandas as pd
import torch.nn as nn

from typing import Tuple
from dotenv import load_dotenv
from deltalake import DeltaTable
from pyarrow.dataset import Dataset
from mlflow.pytorch import load_model
from sklearn.compose import ColumnTransformer


TARGET_COLUMN = "label"

MODEL_PATH='.data'
PREPROCESSOR_PATH='.data/preprocessor.pkl'

# ========================= CONFIGURATION =========================

def load_delta_table():
  LAKEFS_STORAGE_REPO = os.getenv('LAKEFS_STORAGE_REPO')
  LAKEFS_STORAGE_TABLE = os.getenv('LAKEFS_STORAGE_TABLE')
  LAKEFS_STORAGE_BRANCH = os.getenv('LAKEFS_STORAGE_BRANCH')
  
  assert LAKEFS_STORAGE_REPO, 'lakefs storage repo is required'
  assert LAKEFS_STORAGE_TABLE, 'lakefs storage table is required'
  assert LAKEFS_STORAGE_BRANCH, 'lakefs storage branch is required'

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
  
  try:
    DELTA_URI = f"s3://{LAKEFS_STORAGE_REPO}/{LAKEFS_STORAGE_BRANCH}/{LAKEFS_STORAGE_TABLE}"
    dt = DeltaTable(DELTA_URI, storage_options=STORAGE_OPTIONS)
    print(f"Successfully opened Delta table: {DELTA_URI}")
    print(f"Version: {dt.version()}")
  except Exception as e:
    print(f"Failed to open Delta table: {e}")
    sys.exit(1)

  dataset = dt.to_pyarrow_dataset()
  schema = dataset.schema
  return dataset, schema

# ========================= LOAD PREPROCESSOR =========================

def load_preprocessor(path = PREPROCESSOR_PATH) -> ColumnTransformer:
  return joblib.load(path)

# ========================= LOAD MODEL =========================

def load_pytorch_model(path = MODEL_PATH) -> Tuple[nn.Module, torch.device]:
  model = load_model(model_uri=path)
  model.eval()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  model.to(device)

  return model, device

# ========================= GENERATE SAMPLE DATA =========================

def get_sample_batches(ds: Dataset, batch_size: int = 10, max_total_rows: int | None = None):
  scanner = ds.scanner(batch_size=batch_size)
  
  rows_yielded = 0
  for record_batch in scanner.to_batches():
    df = record_batch.to_pandas()
    
    if max_total_rows is not None and rows_yielded + len(df) > max_total_rows:
      # Trim the last batch if we've hit the desired total
      df = df.head(max_total_rows - rows_yielded)
    
    print(f"Yielded batch with {len(df)} rows (total so far: {rows_yielded + len(df):,})")
    yield df
    
    rows_yielded += len(df)
    if max_total_rows is not None and rows_yielded >= max_total_rows:
        break

if __name__ == '__main__':
  load_dotenv()
  
  preprocessor = load_preprocessor()
  model, device = load_pytorch_model()
  
  # -------------------------------------------------------
  # Prepare features (must match train.py behavior)
  # -------------------------------------------------------
  
  dataset, schema = load_delta_table()
  print(f"\nSchema:\n{schema}")
  
  samples = list(get_sample_batches(dataset, batch_size=10, max_total_rows=100))
  samples = pd.concat(samples, ignore_index=True)
  print(f"\nFitted preprocessors on {len(samples):,} sample rows.")

  feature_columns = [c for c in samples.columns if c != TARGET_COLUMN]
  X = preprocessor.transform(samples[feature_columns]).astype(np.float32)

  # -------------------------------------------------------
  # Torch inference
  # -------------------------------------------------------

  X_tensor = torch.from_numpy(X).to(device)
  with torch.no_grad():
    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    # -------------------------------------------------------
    # Attach predictions back to polars DataFrame
    # -------------------------------------------------------

    samples["prediction"] = preds.cpu().numpy()
    samples["confidence"] = probs.max(dim=1).values.cpu().numpy()
    
    print("\nSample predictions:")
    print(samples[["prediction", "confidence", TARGET_COLUMN]].head())
    
    # -------------------------------------------------------
    # Compute and print accuracy
    # -------------------------------------------------------

    correct = (samples["prediction"] == samples[TARGET_COLUMN]).sum()
    total = samples.shape[0]
    accuracy = correct / total if total > 0 else 0.0

    print(f"\nAccuracy on sample dataset: {accuracy:.4f} ({correct}/{total})")
    
    # -------------------------------------------------------
    # Predict for input example
    # -------------------------------------------------------
    
    X_preprocessed = json.load(open('.data/input_example.json', 'r'))
    X_tensor = torch.tensor(X_preprocessed, dtype=torch.float32).to(device)
    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    print("\nPrediction:", preds.item())
    print("Probabilities:", probs[0].tolist())
