"""
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Annotated


# Pydantic v2: use Annotated for list length constraints
BBoxList = Annotated[List[float], Field(min_length=4, max_length=4)]


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BBoxList


class PredictionResponse(BaseModel):
    model_version: str  # "yolov8" or "yolov11"
    num_detections: int
    detections: List[Detection]
    message: str = "OK"
    annotated_image_b64: Optional[str] = None  # Base64-kodiertes JPEG mit Bounding Boxes


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]


class ModelInfoResponse(BaseModel):
    model_version: str
    num_classes: int
    class_names: List[str]
    conf_threshold: float
