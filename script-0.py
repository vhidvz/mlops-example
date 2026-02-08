# Required installations (run in your environment):
# pip install deltalake pyarrow torch torchvision torchaudio mlflow pandas numpy matplotlib scikit-learn

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow.fs import S3FileSystem
from deltalake import DeltaTable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import urllib.parse

# ----------------------------- Configuration -----------------------------
# Placeholders - replace with your actual LakeFS credentials and endpoint
LAKEFS_ENDPOINT_URL = "http://localhost:8000"  # Example: http://localhost:8000 or https://lakefs.yourcompany.com
ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"          # Your LakeFS access key
SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"  # Your LakeFS secret key

REPO = "example-repo"
BRANCH = "main"
TABLE_PATH = "data/table"                     # Path inside the repository/branch to the Delta table
TABLE_URI = f"s3://{REPO}/{BRANCH}/{TABLE_PATH}"

BATCH_SIZE = 1024
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 0.001
HIDDEN_DIMS = [128, 64]
RANDOM_SEED = 42
VAL_SPLIT = 0.2  # 20% of Parquet files held out for validation (file-level split)

# ----------------------------- Setup -----------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Parse endpoint for PyArrow S3FileSystem
parsed = urllib.parse.urlparse(LAKEFS_ENDPOINT_URL)
scheme = parsed.scheme or "http"
host = parsed.netloc

# Create PyArrow S3 filesystem (used for reading individual Parquet files)
s3_fs = S3FileSystem(
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    endpoint_override=host,
    scheme=scheme,
    region="us-east-1",
)

# Storage options for deltalake (S3-compatible access to LakeFS)
storage_options = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": LAKEFS_ENDPOINT_URL,
    "AWS_REGION": "us-east-1",
}
if scheme == "http":
    storage_options["AWS_S3_ALLOW_HTTP"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------- Load Delta Table -----------------------------
try:
    dt = DeltaTable(TABLE_URI, storage_options=storage_options)
    schema_pa = dt.schema().to_pyarrow()
except Exception as e:
    raise RuntimeError(f"Failed to load Delta table from {TABLE_URI}: {e}")

if "label" not in schema_pa.names:
    raise ValueError("Delta table must contain a column named 'label'")

features = [name for name in schema_pa.names if name != "label"]
input_dim = len(features)
print(f"Found {input_dim} feature columns: {features}")

# Get list of underlying Parquet data files and perform file-level train/val split
files = dt.files()
if not files:
    raise ValueError("No data files found in the Delta table")

random.shuffle(files)
split_idx = int((1 - VAL_SPLIT) * len(files))
train_files = files[:split_idx]
val_files = files[split_idx:]

if len(train_files) == 0 or len(val_files) == 0:
    raise ValueError("Not enough Parquet files to create both train and validation sets")

print(f"Split: {len(train_files)} train files, {len(val_files)} validation files")

# Create PyArrow datasets for train/validation (reading only selected files)
train_dataset = ds.dataset(
    train_files,
    filesystem=s3_fs,
    schema=schema_pa,
    format="parquet",
)
val_dataset = ds.dataset(
    val_files,
    filesystem=s3_fs,
    schema=schema_pa,
    format="parquet",
)

# ----------------------------- Preprocessing Statistics (fit on train only) -----------------------------
def compute_preprocessing_stats(dataset):
    """Stream through train dataset to compute mean/std for normalization and label mapping."""
    sums = np.zeros(input_dim, dtype=np.float64)
    sumsqs = np.zeros(input_dim, dtype=np.float64)
    count = 0
    label_set = set()

    scanner = dataset.scanner(use_threads=True)
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if df.empty:
            continue
        x_np = df[features].to_numpy(dtype=np.float64)
        sums += np.nansum(x_np, axis=0)  # nan-safe
        sumsqs += np.nansum(x_np ** 2, axis=0)
        count += x_np.shape[0]
        label_set.update(df["label"].dropna().unique().tolist())

    mean = sums / count
    std = np.sqrt((sumsqs / count) - (mean ** 2) + 1e-8)
    # Handle possible NaN in std (e.g., constant features)
    std = np.nan_to_num(std, nan=1.0)

    unique_labels = sorted(label_set)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    print(f"Computed stats: {count} train rows, {num_classes} classes")
    return mean, std, label_map, num_classes, unique_labels

mean, std, label_map, num_classes, unique_labels = compute_preprocessing_stats(train_dataset)

if num_classes < 2:
    raise ValueError("Classification requires at least 2 classes")

# ----------------------------- Model Definition -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim, HIDDEN_DIMS, num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------- Training Functions -----------------------------
def train_epoch(model, dataset, optimizer, loss_fn, device, mean, std, label_map, batch_size, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    scanner = dataset.scanner(batch_size=batch_size, use_threads=True)
    with torch.enable_grad() if is_train else torch.no_grad():
        for record_batch in scanner.to_batches():
            df = record_batch.to_pandas()
            if df.empty:
                continue

            # Normalize numerical features (assumes all features are numerical)
            x_np = (df[features].values - mean) / std
            x = torch.from_numpy(x_np).float().to(device)

            # Map labels to dense 0..num_classes-1
            y_mapped = df["label"].map(label_map)
            # Drop any rows with missing labels (rare, but safe)
            valid_mask = y_mapped.notna()
            if not valid_mask.all():
                x = x[valid_mask]
                y_mapped = y_mapped[valid_mask]
            y = torch.from_numpy(y_mapped.values).long().to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(x)
            loss = loss_fn(outputs, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total if total > 0 else float("inf")
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def get_predictions(dataset, model, device, mean, std, label_map, batch_size=2048):
    """Collect all predictions on validation set for confusion matrix."""
    model.eval()
    y_true_list = []
    y_pred_list = []

    scanner = dataset.scanner(batch_size=batch_size, use_threads=True)
    with torch.no_grad():
        for record_batch in scanner.to_batches():
            df = record_batch.to_pandas()
            if df.empty:
                continue

            x_np = (df[features].values - mean) / std
            x = torch.from_numpy(x_np).float().to(device)

            y_mapped = df["label"].map(label_map)
            valid_mask = y_mapped.notna()
            if not valid_mask.all():
                x = x[valid_mask]
                y_mapped = y_mapped[valid_mask]

            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true_list.extend(y_mapped[valid_mask].values)
            y_pred_list.extend(preds)

    return np.array(y_true_list), np.array(y_pred_list)

# ----------------------------- MLflow Training Loop -----------------------------
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("hidden_dims", str(HIDDEN_DIMS))
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("max_epochs", MAX_EPOCHS)
    mlflow.log_param("patience", PATIENCE)
    mlflow.log_param("num_train_files", len(train_files))
    mlflow.log_param("num_val_files", len(val_files))
    mlflow.log_param("features", features)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = "best_model.pth"

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_dataset, optimizer, loss_fn, device,
            mean, std, label_map, BATCH_SIZE, is_train=True
        )
        val_loss, val_acc = train_epoch(
            model, val_dataset, optimizer, loss_fn, device,
            mean, std, label_map, BATCH_SIZE, is_train=False
        )

        print(f"Epoch {epoch}/{MAX_EPOCHS} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }, step=epoch)

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  >>> New best model saved")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    # Load best model for final logging
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    # Log final model
    mlflow.pytorch.log_model(model, "model")

    # Confusion matrix artifact (classification only)
    try:
        y_true, y_pred = get_predictions(val_dataset, model, device, mean, std, label_map)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        print("Logged confusion matrix artifact")
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")

print("Training complete. MLflow run finished.")