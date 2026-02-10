"""
curl -X 'POST' \
    'http://127.0.0.1:3000/predict' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "features": [
            0.017, 0.433, 1.045, 2.683, -1.702, -1.001, 0.606, -0.217, 
            0.648, -4.482, 1.154, 1.204, -0.215, 0.356, 0.867, -1.905, 
            -1.445, 1.357, -0.072, 2.199, "green"
        ]
    }'
"""
import os
import torch
import joblib
import dotenv
import uvicorn

import numpy as np
import pandas as pd
import torch.nn as nn

from typing import Tuple
from pydantic import BaseModel
from mlflow.pytorch import load_model
from fastapi import FastAPI, HTTPException
from sklearn.compose import ColumnTransformer


# Load environment variables from .env file
dotenv.load_dotenv()

MODEL_PATH='.data'
PREPROCESSOR_PATH='.data/preprocessor.pkl'

app = FastAPI(
    title="MLOps Example",
    description="Example source code to serve ai model as a service",
    version="0.1.0"
)

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

preprocessor = load_preprocessor()
model, device = load_pytorch_model()

class RequestModel(BaseModel):
    features: list


class ResponseModel(BaseModel):
    prediction: int
    confidence: float


@app.post("/predict", response_model=ResponseModel)
def predict(payload: RequestModel):
    # Ensure features is provided
    if not len(payload.features) == 21:
        raise HTTPException(status_code=400, detail="Invalid input.")

    # Predict the label
    try:
        column_names = [f'num_feature_{i}' for i in range(20)] + ['color']
        sample = pd.DataFrame([payload.features], columns=column_names)
        X = preprocessor.transform(sample).astype(np.float32) # type: ignore
        X_tensor = torch.from_numpy(X).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = int(torch.argmax(probs, dim=1).item())
            confidence = probs[0, prediction].item() # type: ignore
        return ResponseModel(prediction=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", '3000'))

    print("ReDoc OpenAPI: http://0.0.0.0:{}/redoc".format(port))
    print("Swagger OpenAPI: http://0.0.0.0:{}/docs".format(port))

    uvicorn.run(app, host="0.0.0.0", port=port)