import os
import sys

import mlflow

from ml.config import EXPERIMENT_NAME

DEFAULT_TRACKING_URI = "http://localhost:5000"


def configure_mlflow() -> str:
    # MLflow prints emoji in run URLs; the default Windows cp1252 console crashes on them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri
