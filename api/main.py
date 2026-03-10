"""
WasteVision FastAPI – Tag 5 Update

Neu: Prometheus-Metriken fuer Monitoring.
     Custom Metrics: Anzahl Detektionen, Confidence-Scores pro Modell
"""

import base64
import io
import os
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from src.predict import load_model, predict_from_pil, YOLO8_CLASSES, YOLO11_CLASSES
from src.schemas import PredictionResponse, HealthResponse, ModelInfoResponse, Detection

# ─── Custom Prometheus Metrics ────────────────────────────────────────────────
detections_total = Counter(
    "wastevision_detections_total",
    "Gesamtanzahl der erkannten Objekte",
    ["model_version", "class_name"],
)

confidence_histogram = Histogram(
    "wastevision_confidence_score",
    "Verteilung der Confidence-Scores",
    ["model_version"],
    buckets=[0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

requests_per_version = Counter(
    "wastevision_predict_requests_total",
    "Anzahl der Prediction-Requests pro Modellversion",
    ["model_version"],
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

MODEL_REGISTRY = {
    "yolov8":  "models:/wastevision-yolov8@champion",
    "yolov11": "models:/wastevision-yolov11@champion",
}
MODEL_FALLBACK = {
    "yolov8":  os.getenv("MODEL_PATH_V8",  "models/yolo8n_waste.pt"),
    "yolov11": os.getenv("MODEL_PATH_V11", "models/yolo11n_waste.pt"),
}
CLASS_NAMES = {
    "yolov8":  YOLO8_CLASSES,
    "yolov11": YOLO11_CLASSES,
}

app = FastAPI(
    title="WasteVision API",
    description="Waste detection – YOLOv8 (7 Klassen) & YOLOv11 (26 Klassen)",
    version="3.0.0",
)

# Prometheus HTTP-Metriken (Request-Rate, Latenz, Status-Codes)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

_models: dict = {}
_model_source: dict = {}


def get_global_champion() -> str:
    """
    Liest den 'global_champion' Tag aus dem MLflow Experiment.
    Fallback auf 'yolov8' wenn MLflow nicht erreichbar oder Tag nicht gesetzt.
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("wastevision-waste-detection")
        if experiment and "global_champion" in experiment.tags:
            return experiment.tags["global_champion"]
    except Exception:
        pass
    return "yolov8"


def get_model(version: str):
    """
    Lädt Modell aus MLflow Registry (champion alias).
    Fällt auf lokale .pt Datei zurück wenn MLflow nicht erreichbar.
    """
    if version not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unbekannte Version: {version}")

    if version not in _models:
        try:
            import mlflow
            import mlflow.artifacts
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = mlflow.tracking.MlflowClient()
            model_name = f"wastevision-{version}"
            version_info = client.get_model_version_by_alias(model_name, "champion")
            artifact_uri = client.get_model_version_download_uri(
                model_name, version_info.version
            )
            local_dir = mlflow.artifacts.download_artifacts(
                artifact_uri + "/artifacts/yolo_model"
            )
            _models[version] = load_model(local_dir)
            _model_source[version] = f"mlflow:champion (v{version_info.version})"
            print(f"✅ {version} aus MLflow geladen (champion v{version_info.version})")

        except Exception as e:
            fallback_path = MODEL_FALLBACK[version]
            if not os.path.exists(fallback_path):
                raise HTTPException(
                    status_code=503,
                    detail=f"MLflow nicht erreichbar und lokale Datei fehlt: '{fallback_path}'"
                )
            _models[version] = load_model(fallback_path)
            _model_source[version] = f"local:{fallback_path}"
            print(f"⚠️  {version} aus lokaler Datei geladen (MLflow Fallback)")

    return _models[version]


@app.get("/champion")
def champion_info():
    """Gibt die aktuell aktive Champion-Modellversion zurück."""
    version = get_global_champion()
    return {
        "champion_version": version,
        "num_classes": len(CLASS_NAMES[version]),
        "class_names": CLASS_NAMES[version],
    }


@app.get("/")
def root():
    return {
        "message": "Welcome to WasteVision API 🗑️",
        "docs": "/docs",
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded=[
            f"{v} ({_model_source.get(v, 'not loaded')})"
            for v in _models
        ],
    )


@app.get("/models/{version}", response_model=ModelInfoResponse)
def model_info(
    version: str,
    conf_threshold: float = Query(default=0.45, ge=0.0, le=1.0),
):
    if version not in CLASS_NAMES:
        raise HTTPException(status_code=404, detail=f"Unbekannte Version: {version}")
    return ModelInfoResponse(
        model_version=version,
        num_classes=len(CLASS_NAMES[version]),
        class_names=CLASS_NAMES[version],
        conf_threshold=conf_threshold,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_champion(
    file: UploadFile = File(...),
    conf_threshold: float = Query(default=0.45, ge=0.0, le=1.0),
):
    """Sendet das Bild an das aktuell aktive Champion-Modell."""
    version = get_global_champion()
    return await predict(version, file, conf_threshold)


@app.post("/predict/{version}", response_model=PredictionResponse)
async def predict(
    version: str,
    file: UploadFile = File(...),
    conf_threshold: float = Query(default=0.45, ge=0.0, le=1.0),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Nur jpg, png, webp unterstützt.")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Bild konnte nicht gelesen werden.")

    model = get_model(version)
    result = predict_from_pil(model, pil_image, conf_threshold=conf_threshold)
    detections = [Detection(**d) for d in result["detections"]]

    # Annotiertes Bild (BGR numpy array von YOLO) als Base64-JPEG kodieren
    annotated_bgr = result["annotated_image"]
    annotated_pil = Image.fromarray(annotated_bgr[..., ::-1])  # BGR → RGB
    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=85)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode()

    # Custom Metriken aktualisieren
    requests_per_version.labels(model_version=version).inc()
    for det in result["detections"]:
        detections_total.labels(
            model_version=version,
            class_name=det["class_name"],
        ).inc()
        confidence_histogram.labels(model_version=version).observe(det["confidence"])

    return PredictionResponse(
        model_version=version,
        num_detections=result["num_detections"],
        detections=detections,
        message="Erkennung erfolgreich" if detections else "Kein Müll über Schwellwert erkannt",
        annotated_image_b64=annotated_b64,
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)