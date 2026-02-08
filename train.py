import os
import sys
import torch
import mlflow

from dotenv import load_dotenv
from deltalake import DeltaTable


def init_mlflow():
  MLFLOW_SERVER_ENDPOINT_URL = os.getenv('MLFLOW_SERVER_ENDPOINT_URL')
  assert MLFLOW_SERVER_ENDPOINT_URL, 'server endpoint url is required'
  mlflow.set_tracking_uri(MLFLOW_SERVER_ENDPOINT_URL)

def load_delta_table():
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


if __name__ == '__main__':
  load_dotenv()
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  dataset, schema = load_delta_table()
  print(f"Schema: {schema}")

  