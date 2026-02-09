"""
Complete, runnable Python script for streaming a Delta Lake table from LakeFS,
preprocessing, training a PyTorch MLP (classification), and tracking with MLflow.

Key features:
- Streams data in batches using deltalake + PyArrow (never loads full dataset into RAM).
- Simple but effective train/val split via batch index (deterministic, ~80/20).
- Preprocessing: fits StandardScaler + OneHotEncoder on a small sample, then transforms batches.
- PyTorch MLP for classification (assumes "label" column; works with string or int labels).
- Full MLflow tracking: params, per-epoch metrics, final model, confusion matrix artifact.
- Early stopping, device handling (CPU/GPU), batch size config.
- Error handling and detailed comments.
"""
import os
import sys
import torch
import joblib
import mlflow

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from deltalake import DeltaTable
from typing import Iterator, Tuple
from pyarrow.dataset import Dataset
from sklearn.compose import ColumnTransformer
from mlflow.models.signature import infer_signature
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


TARGET_COLUMN = "label"
VAL_SPLIT_MOD = 5 # every 5th PyArrow batch goes to validation (≈20%)

MLFLOW_EXPERIMENT_NAME = "delta_lake_mlp_training"
MLFLOW_REGISTERED_MODEL_NAME="MLPClassifier"

PATIENCE = 5 # early stopping patience
NUM_EPOCHS = 20
BATCH_SIZE = 256
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001

# ========================= CONFIGURATION =========================

def mlflow_tracking():
  MLFLOW_SERVER_ENDPOINT_URL = os.getenv('MLFLOW_SERVER_ENDPOINT_URL')
  assert MLFLOW_SERVER_ENDPOINT_URL, 'server endpoint url is required'
  mlflow.set_tracking_uri(MLFLOW_SERVER_ENDPOINT_URL)
  mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

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

# ========================= PREPROCESSING (fit on sample) =========================

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

# ========================= PYTORCH MODEL =========================

class TabularMLP(nn.Module):
  def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.2):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_size, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

# ========================= STREAMING DATASET =========================

class DeltaStreamingDataset(IterableDataset):
    """IterableDataset that streams batches from Delta Lake / PyArrow."""

    def __init__(
        self,
        pa_dataset: Dataset,
        preprocessor,
        label_encoder,
        batch_size: int,
        is_train: bool,
        feature_columns,
        val_split_mod: int = 5,
    ):
        self.pa_dataset = pa_dataset
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.is_train = is_train
        self.feature_columns = feature_columns
        self.val_split_mod = val_split_mod

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # New scanner every epoch (full pass through data)
        scanner = self.pa_dataset.scanner(batch_size=self.batch_size)

        for batch_idx, batch in enumerate(scanner.to_batches()):
            # Deterministic split based on batch index (no extra memory)
            is_val_batch = (batch_idx % self.val_split_mod) == 0

            if (self.is_train and not is_val_batch) or (not self.is_train and is_val_batch):
                df = batch.to_pandas()

                # Preprocess
                X_np = self.preprocessor.transform(df[self.feature_columns])
                y_np = self.label_encoder.transform(df[TARGET_COLUMN])

                X_t = torch.from_numpy(X_np).float()
                y_t = torch.from_numpy(y_np).long()

                yield X_t, y_t


if __name__ == '__main__':
  load_dotenv()

  dataset, schema = load_delta_table()
  print(f"\nSchema:\n{schema}")

  if TARGET_COLUMN not in schema.names:
      raise ValueError(f"Target column '{TARGET_COLUMN}' not found in table.")
  feature_columns = [col for col in schema.names if col != TARGET_COLUMN]

  sample_dfs = list(get_sample_batches(dataset, batch_size=10, max_total_rows=1000))
  sample_df = pd.concat(sample_dfs, ignore_index=True)
  print(f"\nFitted preprocessors on {len(sample_df):,} sample rows.")

  # Identify numerical vs categorical columns
  numerical_cols = []
  categorical_cols = []
  for col in feature_columns:
      if pd.api.types.is_numeric_dtype(sample_df[col]):
          numerical_cols.append(col)
      else:
          categorical_cols.append(col)
  print(f"Numerical features: {numerical_cols}")
  print(f"Categorical features: {categorical_cols}")

  # Preprocessor pipeline
  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), numerical_cols),
          ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
      ],
      remainder="drop",
      verbose_feature_names_out=False,
  )

  preprocessor.fit(sample_df[feature_columns])

  # Transform sample to determine input dimension
  sample_X = preprocessor.transform(sample_df[feature_columns]).astype(np.float32)
  input_size = sample_X.shape[1]
  print(f"Input feature size after preprocessing: {input_size}")

  # Label encoder (supports string or numeric labels)
  label_enc = LabelEncoder()
  label_enc.fit(sample_df[TARGET_COLUMN])
  num_classes = len(label_enc.classes_)
  print(f"Number of classes: {num_classes} → {label_enc.classes_}")

  model = TabularMLP(input_size, HIDDEN_SIZE, num_classes)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  model.to(device)
  
  # Create loaders (batch_size=None because we already batch in the iterator)
  train_dataset = DeltaStreamingDataset(
      dataset, preprocessor, label_enc, BATCH_SIZE,
      is_train=True, val_split_mod=VAL_SPLIT_MOD,
      feature_columns=feature_columns
  )
  val_dataset = DeltaStreamingDataset(
      dataset, preprocessor, label_enc, BATCH_SIZE,
      is_train=False, val_split_mod=VAL_SPLIT_MOD,
      feature_columns=feature_columns
  )
  
  pin_memory = torch.cuda.is_available() # or (device.type == 'cuda')
  train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0, pin_memory=pin_memory)
  val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0, pin_memory=pin_memory)

  # ========================= TRAINING LOOP WITH MLFLOW =========================
  
  mlflow_tracking()
  with mlflow.start_run(run_name="lakefs_delta_mlp") as run:
    # Log hyperparameters
    mlflow.log_params({
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "hidden_size": HIDDEN_SIZE,
        "patience": PATIENCE,
        "val_split_mod": VAL_SPLIT_MOD,
        "device": str(device),
    })

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    all_preds_best = []
    all_labels_best = []
    for epoch in range(NUM_EPOCHS):
      # === Training ===
      model.train()
      
      train_loss = 0.0
      train_batches = 0
      for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

      avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0

      # === Validation ===
      model.eval()
      
      total = 0
      correct = 0
      val_loss = 0.0
      val_batches = 0
      all_preds = []
      all_labels = []
      with torch.no_grad():
        for X_batch, y_batch in val_loader:
          X_batch, y_batch = X_batch.to(device), y_batch.to(device)

          outputs = model(X_batch)
          loss = criterion(outputs, y_batch)

          val_batches += 1
          val_loss += loss.item()

          _, predicted = torch.max(outputs, 1)
          total += y_batch.size(0)
          correct += (predicted == y_batch).sum().item()

          all_preds.extend(predicted.cpu().numpy())
          all_labels.extend(y_batch.cpu().numpy())

      val_accuracy = correct / total if total > 0 else 0.0
      avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

      # Log metrics
      mlflow.log_metrics({
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy,
      }, step=epoch)

      print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
      )

      # Early stopping
      if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          best_epoch = epoch
          patience_counter = 0
          torch.save(model.state_dict(), "best_model.pth")
          all_preds_best, all_labels_best = all_preds, all_labels
      else:
          patience_counter += 1
          if patience_counter >= PATIENCE:
              print(f"Early stopping triggered at epoch {epoch+1}")
              break

    # ========================= LOAD BEST MODEL FOR LOGGING =========================
    print("Loading best model state dict for final logging...")
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("Best model loaded.")
    else:
        print("No best model checkpoint found; logging current model.")

    # ========================= INFER SIGNATURE AND INPUT EXAMPLE =========================
    model.eval()
    sample_input_np = sample_X[:1].copy()  # Small real preprocessed batch
    sample_tensor = torch.from_numpy(sample_input_np).float().to(device)
    with torch.no_grad():
        sample_predictions = model(sample_tensor).cpu().numpy()

    signature = infer_signature(sample_input_np, sample_predictions)
    input_example = sample_input_np

    # ========================= FINAL ARTIFACTS =========================
    print("Logging model and artifacts to MLflow...")

    # Log the trained model with signature and input example (eliminates warning)
    mlflow.pytorch.log_model(
      pytorch_model=model,
      name="model",
      signature=signature,
      input_example=input_example,
      registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
    )

    # Confusion matrix (from the best validation pass)
    cm = confusion_matrix(all_labels_best, all_preds_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    # log preprocessor (as pickle) if you want to reuse it
    joblib.dump(preprocessor, "preprocessor.pkl")
    mlflow.log_artifact("preprocessor.pkl")

    print(f"Training completed! Run ID: {run.info.run_id}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
