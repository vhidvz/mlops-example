import torch
import joblib

import torch.nn as nn

from mlflow.pytorch import load_model
from sklearn.compose import ColumnTransformer


MODEL_PATH='.data'
PREPROCESSOR_PATH='.data/preprocessor.pkl'

# ========================= LOAD PREPROCESSOR =========================

def load_preprocessor(path = PREPROCESSOR_PATH) -> ColumnTransformer:
  return joblib.load(path)


# ========================= LOAD MODEL =========================

def load_pytorch_model(path = MODEL_PATH) -> nn.Module:
  model = load_model(model_uri=path)
  model.eval()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  model.to(device)

  return model

if __name__ == '__main__':
  load_preprocessor()
  load_pytorch_model()