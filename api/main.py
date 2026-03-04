
"""
WasteVision FastAPI – serves YOLOv8 (7 classes) and YOLOv11 (26 classes).

Endpoints:
  GET  /              → welcome
  GET  /health        → health check
  GET  /models/{ver}  → model info
  POST /predict/{ver} → run inference on uploaded image
"""

import io
import os
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

from src.predict import load_model, predict_from_pil, YOLO8_CLASSES, YOLO11_CLASSES
from src.schemas import PredictionResponse, HealthResponse, ModelInfoResponse, Detection

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="WasteVision API",
    description="Waste detection using YOLOv8 (7 classes) and YOLOv11 (26 classes)",
    version="1.0.0",
)

# Model paths – override via environment variables in production
MODEL_PATHS = {
    "yolov8": os.getenv("MODEL_PATH_V8", "models/yolo8n_waste.pt"),
    "yolov11": os.getenv("MODEL_PATH_V11", "models/yolo11n_waste.pt"),
}

CLASS_NAMES = {
    "yolov8": YOLO8_CLASSES,
    "yolov11": YOLO11_CLASSES,
}

# Lazy-loaded model cache
_models: dict = {}


def get_model(version: str):
    """Load model on first request and cache it."""
    if version not in MODEL_PATHS:
        raise HTTPException(status_code=404, detail=f"Unknown model version: {version}. Use 'yolov8' or 'yolov11'.")
    if version not in _models:
        path = MODEL_PATHS[version]
        if not os.path.exists(path):
            raise HTTPException(
                status_code=503,
                detail=f"Model file not found: '{path}'. Place your .pt file there or set the env var."
            )
        _models[version] = load_model(path)
    return _models[version]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Welcome")
def root():
    return {"message": "Welcome to WasteVision API 🗑️", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    return HealthResponse(
        status="ok",
        models_loaded=list(_models.keys()),
    )


@app.get("/models/{version}", response_model=ModelInfoResponse, summary="Model info")
def model_info(
    version: str,
    conf_threshold: float = Query(default=0.45, ge=0.0, le=1.0),
):
    """Returns class names and config for the requested model version."""
    if version not in CLASS_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown model: {version}")
    return ModelInfoResponse(
        model_version=version,
        num_classes=len(CLASS_NAMES[version]),
        class_names=CLASS_NAMES[version],
        conf_threshold=conf_threshold,
    )


@app.post("/predict/{version}", response_model=PredictionResponse, summary="Run waste detection")
async def predict(
    version: str,
    file: UploadFile = File(..., description="Image file (jpg, png, webp)"),
    conf_threshold: float = Query(default=0.45, ge=0.0, le=1.0, description="Confidence threshold"),
):
    """
    Upload an image and get waste detection results.

    - **version**: `yolov8` (7 classes) or `yolov11` (26 classes)
    - **file**: image file to analyse
    - **conf_threshold**: minimum confidence to include a detection (default 0.45)
    """
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported image type. Use jpg, png or webp.")

    # Read and convert image
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # Run inference
    model = get_model(version)
    result = predict_from_pil(model, pil_image, conf_threshold=conf_threshold)

    detections = [Detection(**d) for d in result["detections"]]

    return PredictionResponse(
        model_version=version,
        num_detections=result["num_detections"],
        detections=detections,
        message="Detection successful" if detections else "No waste detected above threshold",
    )


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
