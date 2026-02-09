import os
import sys
import mlflow

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts


MLFLOW_METADATA_PATH='.data/registered_model_meta'
MLFLOW_EXPERIMENT_NAME = "delta_lake_mlp_training"
MLFLOW_REGISTERED_MODEL_NAME="MLPClassifier"


def mlflow_tracking():
  MLFLOW_SERVER_ENDPOINT_URL = os.getenv('MLFLOW_SERVER_ENDPOINT_URL')
  assert MLFLOW_SERVER_ENDPOINT_URL, 'server endpoint url is required'
  mlflow.set_tracking_uri(MLFLOW_SERVER_ENDPOINT_URL)
  mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


if __name__ == '__main__':
  load_dotenv()
  mlflow_tracking()
  
  client = MlflowClient()
  versions = client.search_model_versions(
    f"name='{MLFLOW_REGISTERED_MODEL_NAME}'")

  best_val_accuracy = 0
  best_version = versions[0]
  for v in versions:
    run = client.get_run(v.run_id) if v.run_id else None
    if run and run.data.metrics['val_accuracy'] > best_val_accuracy:
      best_version, best_val_accuracy = v, run.data.metrics['val_accuracy']
  print(f"{best_version.name}:v{best_version.version} model selected.")
  
  # Download Preprocessor
  local_path = client.download_artifacts(
    run_id=best_version.run_id, # type: ignore
    path="preprocessor.pkl",
    dst_path='.data',
  )
  print("Preprocessor downloaded to:", local_path)
  
  # Download model
  model_uri = f"models:/{best_version.name}/{best_version.version}"
  local_path = download_artifacts(
    artifact_uri=model_uri,
    dst_path='.data/'
  )
  print("Model downloaded to:", local_path)

  if not os.path.exists(MLFLOW_METADATA_PATH):
    print('\nloading model failed, check ".data" directory or logs.')
    sys.exit(1)
    
  with open(MLFLOW_METADATA_PATH, 'r') as f:
    print(f"\n{f.read()}")