"""
Tests for Tag 1:
  - predict.py: unit tests (no real model needed – mocked)
  - api/main.py: integration tests via FastAPI TestClient
"""

import io
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_dummy_image_bytes(width=100, height=100) -> bytes:
    """Creates a small white JPEG in memory."""
    img = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 200)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def make_mock_model(class_names: dict, num_boxes: int = 2):
    """Returns a MagicMock that mimics ultralytics YOLO inference output."""
    model = MagicMock()

    boxes = []
    for i in range(num_boxes):
        box = MagicMock()
        box.cls.item.return_value = i % len(class_names)
        box.conf.item.return_value = 0.85
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = [10.0, 20.0, 80.0, 90.0]
        boxes.append(box)

    result = MagicMock()
    result.boxes = boxes
    result.names = class_names
    result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    model.return_value = [result]
    return model


# ─── Unit tests: predict.py ───────────────────────────────────────────────────

class TestPredictImage:
    """Tests for src/predict.py – no real model weights needed."""

    def test_predict_returns_correct_keys(self):
        from src.predict import predict_image
        mock_model = make_mock_model({0: "plastic", 1: "glass"}, num_boxes=2)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = predict_image(mock_model, img)

        assert "detections" in result
        assert "annotated_image" in result
        assert "num_detections" in result

    def test_predict_num_detections(self):
        from src.predict import predict_image
        mock_model = make_mock_model({0: "plastic", 1: "glass"}, num_boxes=3)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = predict_image(mock_model, img)
        assert result["num_detections"] == 3

    def test_detection_has_required_fields(self):
        from src.predict import predict_image
        mock_model = make_mock_model({0: "cardboard"}, num_boxes=1)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = predict_image(mock_model, img)
        det = result["detections"][0]

        assert "class_id" in det
        assert "class_name" in det
        assert "confidence" in det
        assert "bbox" in det
        assert len(det["bbox"]) == 4

    def test_confidence_in_valid_range(self):
        from src.predict import predict_image
        mock_model = make_mock_model({0: "metal"}, num_boxes=2)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = predict_image(mock_model, img)
        for det in result["detections"]:
            assert 0.0 <= det["confidence"] <= 1.0

    def test_predict_from_pil_accepts_pil_image(self):
        from src.predict import predict_from_pil
        mock_model = make_mock_model({0: "paper"}, num_boxes=1)
        pil_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        result = predict_from_pil(mock_model, pil_img)
        assert result["num_detections"] == 1

    def test_load_model_raises_on_missing_file(self):
        from src.predict import load_model
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_path/model.pt")


# ─── Integration tests: FastAPI ───────────────────────────────────────────────

class TestAPI:
    """Tests for api/main.py using FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def patch_models(self):
        """Patch get_model so no real .pt file is needed."""
        mock_model = make_mock_model({0: "plastic", 1: "glass"}, num_boxes=2)
        with patch("api.main.get_model", return_value=mock_model):
            yield

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "WasteVision" in response.json()["message"]

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_model_info_yolov8(self, client):
        response = client.get("/models/yolov8")
        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == 7
        assert "plastic" in data["class_names"]

    def test_model_info_yolov11(self, client):
        response = client.get("/models/yolov11")
        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == 26

    def test_model_info_unknown_version(self, client):
        response = client.get("/models/yolov999")
        assert response.status_code == 404

    def test_predict_valid_image(self, client):
        image_bytes = make_dummy_image_bytes()
        response = client.post(
            "/predict/yolov8",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "num_detections" in data
        assert data["model_version"] == "yolov8"

    def test_predict_unsupported_format(self, client):
        response = client.post(
            "/predict/yolov8",
            files={"file": ("test.gif", b"GIF89a", "image/gif")},
        )
        assert response.status_code == 415

    def test_predict_with_conf_threshold(self, client):
        image_bytes = make_dummy_image_bytes()
        response = client.post(
            "/predict/yolov8?conf_threshold=0.9",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
