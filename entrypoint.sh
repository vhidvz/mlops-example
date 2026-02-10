#!/bin/sh
set -e
exec fastapi run --workers "$WORKERS" app.py --port "$PORT"
