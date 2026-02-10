import json
import torch
import joblib
import random

import numpy as np
import polars as pl
import torch.nn as nn

from typing import Tuple
from mlflow.pytorch import load_model
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification


TARGET_COLUMN = "label"

MODEL_PATH='.data'
PREPROCESSOR_PATH='.data/preprocessor.pkl'

RANDOM_SEED = 1234
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

def sample_generation(n_samples = 1_000) -> pl.DataFrame:
  print("Generating synthetic dataset...")

  X, y = make_classification(
    n_samples=n_samples,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=2,
    random_state=RANDOM_SEED
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

if __name__ == '__main__':
  preprocessor = load_preprocessor()
  model, device = load_pytorch_model()
  
  # -------------------------------------------------------
  # Prepare features (must match train.py behavior)
  # -------------------------------------------------------
  samples = sample_generation()

  pdf = samples.to_pandas()
  feature_columns = [c for c in pdf.columns if c != TARGET_COLUMN]

  X = preprocessor.transform(pdf[feature_columns]).astype(np.float32) # type: ignore

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

    samples = samples.with_columns([
      pl.Series("prediction", preds.cpu().numpy()),
      pl.Series("confidence", probs.max(dim=1).values.cpu().numpy()),
    ])

    print("\nPredictions:")
    print(samples.select(["prediction", "confidence"]).head())

    # -------------------------------------------------------
    # Compute and print accuracy
    # -------------------------------------------------------

    print(samples[TARGET_COLUMN])
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
