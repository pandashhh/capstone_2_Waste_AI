#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# MLflow Server starten für WasteVision
#
# OPTION A – Komplett lokal (kein GCS, gut zum Testen):
#   ./start_mlflow.sh local
#
# OPTION B – Mit GCS als Artifact Store (für Produktion):
#   export GCS_BUCKET=dein-bucket-name
#   export GOOGLE_APPLICATION_CREDENTIALS=/pfad/zu/sa_key.json
#   ./start_mlflow.sh gcs
# ─────────────────────────────────────────────────────────────────────────────

MODE=${1:-local}
MLFLOW_PORT=5001
BACKEND_STORE="sqlite:///mlflow_store/mlflow.db"

mkdir -p mlflow_store

if [ "$MODE" = "local" ]; then
    echo "🚀 Starte MLflow lokal (SQLite Backend, lokale Artifacts)"
    mlflow server \
        --host 0.0.0.0 \
        --port $MLFLOW_PORT \
        --backend-store-uri $BACKEND_STORE \
        --default-artifact-root ./mlflow_store/artifacts

elif [ "$MODE" = "gcs" ]; then
    if [ -z "$GCS_BUCKET" ]; then
        echo "❌ Bitte GCS_BUCKET setzen: export GCS_BUCKET=dein-bucket-name"
        exit 1
    fi
    if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "❌ Bitte GOOGLE_APPLICATION_CREDENTIALS setzen"
        exit 1
    fi
    echo "🚀 Starte MLflow mit GCS Artifact Store: gs://$GCS_BUCKET/mlflow-artifacts"
    mlflow server \
        --host 0.0.0.0 \
        --port $MLFLOW_PORT \
        --backend-store-uri $BACKEND_STORE \
        --default-artifact-root gs://$GCS_BUCKET/mlflow-artifacts

else
    echo "❌ Unbekannter Modus: $MODE. Nutze 'local' oder 'gcs'"
    exit 1
fi
