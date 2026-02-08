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

Installation (run once):
pip install deltalake pyarrow pandas torch torchvision torchaudio mlflow scikit-learn matplotlib

For GPU: add --index-url https://download.pytorch.org/whl/cu121 (adjust for your CUDA version).
"""

import os
import sys
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

import mlflow
import mlflow.pytorch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from deltalake import DeltaTable


# ========================= CONFIGURATION =========================

# LakeFS / S3-compatible settings (replace with your values)
LAKEFS_ENDPOINT = "http://your-lakefs-host:8000"          # e.g. http://localhost:8000 or your S3 gateway URL
LAKEFS_ACCESS_KEY = "your-lakefs-access-key"
LAKEFS_SECRET_KEY = "your-lakefs-secret-key"

REPO = "example-repo"
BRANCH = "main"
TABLE_PATH = "data/table"                                 # relative path inside the branch

# Delta Lake URI (LakeFS treats repo/branch as the "bucket")
DELTA_URI = f"s3://{REPO}/{BRANCH}/{TABLE_PATH}"

STORAGE_OPTIONS = {
    "AWS_ENDPOINT_URL": LAKEFS_ENDPOINT,
    "AWS_ACCESS_KEY_ID": LAKEFS_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": LAKEFS_SECRET_KEY,
    "AWS_REGION": "us-east-1",                # often required; can be dummy value
    "AWS_S3_FORCE_PATH_STYLE": "true",        # critical for LakeFS / MinIO-style endpoints
    # "AWS_S3_ALLOW_UNSAFE_RENAME": "true",   # only needed if you also write
}

# Training hyperparameters
BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
PATIENCE = 5                     # early stopping patience
VAL_SPLIT_MOD = 5                # every 5th PyArrow batch goes to validation (≈20%)

MLFLOW_TRACKING_URI = "http://localhost:5000"   # or "file:///path/to/mlruns" for local
MLFLOW_EXPERIMENT_NAME = "delta_lake_mlp_training"

TARGET_COLUMN = "label"

# ========================= OPEN DELTA TABLE =========================

try:
    dt = DeltaTable(DELTA_URI, storage_options=STORAGE_OPTIONS)
    print(f"✅ Successfully opened Delta table: {DELTA_URI}")
    print(f"   Version: {dt.version()}")
except Exception as e:
    print(f"❌ Failed to open Delta table: {e}")
    print("   Check LakeFS credentials, endpoint, and that the path exists.")
    sys.exit(1)

dataset = dt.to_pyarrow_dataset()
schema = dataset.schema
print(f"   Schema: {schema}")

if TARGET_COLUMN not in schema.names:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in table.")

feature_columns = [col for col in schema.names if col != TARGET_COLUMN]


# ========================= PREPROCESSING (fit on sample) =========================

# Take a small sample to fit scalers/encoders (memory-safe)
def get_sample_batches(ds: pa.dataset.Dataset, n_batches: int = 10) -> pd.DataFrame:
    batches = []
    for i, batch in enumerate(ds.to_batches()):
        if i >= n_batches:
            break
        batches.append(batch.to_pandas())
    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()

sample_df = get_sample_batches(dataset, n_batches=10)
print(f"   Fitted preprocessors on {len(sample_df):,} sample rows.")

# Identify numerical vs categorical columns
numerical_cols = []
categorical_cols = []
for col in feature_columns:
    if pd.api.types.is_numeric_dtype(sample_df[col]):
        numerical_cols.append(col)
    else:
        categorical_cols.append(col)

print(f"   Numerical features: {numerical_cols}")
print(f"   Categorical features: {categorical_cols}")

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
sample_X = preprocessor.transform(sample_df[feature_columns])
input_size = sample_X.shape[1]
print(f"   Input feature size after preprocessing: {input_size}")

# Label encoder (supports string or numeric labels)
label_enc = LabelEncoder()
label_enc.fit(sample_df[TARGET_COLUMN])
num_classes = len(label_enc.classes_)
print(f"   Number of classes: {num_classes} → {label_enc.classes_}")


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


model = TabularMLP(input_size, HIDDEN_SIZE, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"   Model moved to {device}")


# ========================= STREAMING DATASET =========================

class DeltaStreamingDataset(IterableDataset):
    """IterableDataset that streams batches from Delta Lake / PyArrow."""

    def __init__(
        self,
        pa_dataset: pa.dataset.Dataset,
        preprocessor,
        label_encoder,
        batch_size: int,
        is_train: bool,
        val_split_mod: int = 5,
    ):
        self.pa_dataset = pa_dataset
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.is_train = is_train
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
                X_np = self.preprocessor.transform(df[feature_columns])
                y_np = self.label_encoder.transform(df[TARGET_COLUMN])

                X_t = torch.from_numpy(X_np).float()
                y_t = torch.from_numpy(y_np).long()

                yield X_t, y_t


# Create loaders (batch_size=None because we already batch in the iterator)
train_dataset = DeltaStreamingDataset(
    dataset, preprocessor, label_enc, BATCH_SIZE, is_train=True, val_split_mod=VAL_SPLIT_MOD
)
val_dataset = DeltaStreamingDataset(
    dataset, preprocessor, label_enc, BATCH_SIZE, is_train=False, val_split_mod=VAL_SPLIT_MOD
)

train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0, pin_memory=True)


# ========================= TRAINING LOOP WITH MLFLOW =========================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

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
        val_loss = 0.0
        val_batches = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                val_batches += 1

                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        val_accuracy = correct / total if total > 0 else 0.0

        # Log metrics
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
        }, step=epoch)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            # Optional: torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # ========================= FINAL ARTIFACTS =========================
    print("Logging model and artifacts to MLflow...")

    # Log the trained model
    mlflow.pytorch.log_model(model, "model")

    # Confusion matrix (from last validation pass)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    # Optional: log preprocessor (as pickle) if you want to reuse it
    import joblib
    joblib.dump(preprocessor, "preprocessor.pkl")
    mlflow.log_artifact("preprocessor.pkl")

    print(f"✅ Training completed! Run ID: {run.info.run_id}")
    print(f"   Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")