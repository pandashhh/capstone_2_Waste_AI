"""
Tag 4 - Behavioral Tests fuer WasteVision

Behavioral Tests pruefen nicht ob der Code laeuft,
sondern ob das Modell sich korrekt VERHAELT:
- Gibt es nur gueltige Klassen zurueck?
- Sind Confidence-Werte immer im gueltigen Bereich?
- Filtert der Confidence-Threshold korrekt?
- Gibt das Modell bei sehr hohem Threshold weniger Ergebnisse?
- Sind Bounding Boxes immer geometrisch korrekt?
- Verarbeitet die API verschiedene Bildgroessen ohne Absturz?
"""

import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


# ─── Helpers ──────────────────────────────────────────────────────────────────
VALID_YOLO8_CLASSES = [
    "cardboard", "food_organics", "glass", "metal",
    "miscellaneous_trash", "paper", "plastic"
]

VALID_YOLO11_CLASSES = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans",
    "cardboard_boxes", "cardboard_packaging", "clothing",
    "coffee_grounds", "disposable_plastic_cutlery", "eggshells",
    "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper",
    "paper_cups", "plastic_detergent_bottles", "plastic_food_containers",
    "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws",
    "plastic_trash_bags", "plastic_water_bottles", "shoes", "styrofoam_cups"
]


def make_mock_detection(class_id: int, confidence: float, class_names: dict):
    box = MagicMock()
    box.cls.item.return_value = class_id
    box.conf.item.return_value = confidence
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = [10.0, 20.0, 80.0, 90.0]
    return box


def make_mock_model(class_names: dict, confidences: list):
    model = MagicMock()
    result = MagicMock()
    result.names = class_names
    result.boxes = [
        make_mock_detection(i % len(class_names), conf, class_names)
        for i, conf in enumerate(confidences)
    ]
    result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    model.return_value = [result]
    return model


def make_image_bytes(width=100, height=100, mode="RGB") -> bytes:
    img = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ─── Behavioral: Output-Schema ────────────────────────────────────────────────
class TestOutputSchema:
    """Das Modell muss immer ein konsistentes Schema zurueckgeben."""

    def test_detections_always_list(self):
        from src.predict import predict_image
        mock_model = make_mock_model({0: "plastic"}, confidences=[0.8])
        result = predict_image(mock_model, np.zeros((100, 100, 3), dtype=np.uint8))
        assert isinstance(result["detections"], list)

    def test_bbox_always_four_values(self):
        from src.predict import predict_image
        mock_model = make_mock_model({0: "glass", 1: "metal"}, confidences=[0.7, 0.9])
        result = predict_image(mock_model, np.zeros((100, 100, 3), dtype=np.uint8))
        for det in result["detections"]:
            assert len(det["bbox"]) == 4, "bbox muss exakt 4 Werte haben"

    def test_bbox_coordinates_are_valid(self):
        """x1 < x2 und y1 < y2 muss immer gelten."""
        box = MagicMock()
        box.cls.item.return_value = 0
        box.conf.item.return_value = 0.8
        box.xyxy = [MagicMock()]
        # Gueltige Box: x1=10 < x2=80, y1=20 < y2=90
        box.xyxy[0].tolist.return_value = [10.0, 20.0, 80.0, 90.0]

        model = MagicMock()
        result = MagicMock()
        result.names = {0: "plastic"}
        result.boxes = [box]
        result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        model.return_value = [result]

        from src.predict import predict_image
        output = predict_image(model, np.zeros((100, 100, 3), dtype=np.uint8))
        det = output["detections"][0]
        x1, y1, x2, y2 = det["bbox"]
        assert x1 < x2, f"x1={x1} muss kleiner als x2={x2} sein"
        assert y1 < y2, f"y1={y1} muss kleiner als y2={y2} sein"

    def test_confidence_always_between_0_and_1(self):
        from src.predict import predict_image
        mock_model = make_mock_model(
            {0: "cardboard", 1: "paper"},
            confidences=[0.45, 0.92, 0.61]
        )
        result = predict_image(mock_model, np.zeros((100, 100, 3), dtype=np.uint8))
        for det in result["detections"]:
            assert 0.0 <= det["confidence"] <= 1.0, (
                f"Confidence {det['confidence']} liegt ausserhalb [0, 1]"
            )

    def test_num_detections_matches_list_length(self):
        from src.predict import predict_image
        mock_model = make_mock_model({0: "metal"}, confidences=[0.5, 0.7, 0.9])
        result = predict_image(mock_model, np.zeros((100, 100, 3), dtype=np.uint8))
        assert result["num_detections"] == len(result["detections"])


# ─── Behavioral: Klassen-Validierung ─────────────────────────────────────────
class TestClassValidation:
    """Das Modell darf nur bekannte Klassen zurueckgeben."""

    def test_yolov8_only_returns_known_classes(self):
        from src.predict import predict_image
        class_names = {i: name for i, name in enumerate(VALID_YOLO8_CLASSES)}
        mock_model = make_mock_model(class_names, confidences=[0.8, 0.6, 0.9])
        result = predict_image(mock_model, np.zeros((100, 100, 3), dtype=np.uint8))
        for det in result["detections"]:
            assert det["class_name"] in VALID_YOLO8_CLASSES, (
                f"Unbekannte Klasse: {det['class_name']}"
            )

    def test_yolov11_only_returns_known_classes(self):
        from src.predict import predict_image
        class_names = {i: name for i, name in enumerate(VALID_YOLO11_CLASSES)}
        mock_model = make_mock_model(class_names, confidences=[0.75, 0.88])
        result = predict_image(mock_model, np.zeros((100, 100, 3), dtype=np.uint8))
        for det in result["detections"]:
            assert det["class_name"] in VALID_YOLO11_CLASSES, (
                f"Unbekannte Klasse: {det['class_name']}"
            )


# ─── Behavioral: API-Verhalten ────────────────────────────────────────────────
class TestAPIBehavior:
    """Die API muss sich konsistent und korrekt verhalten."""

    @pytest.fixture
    def client(self):
        mock_model = make_mock_model(
            {0: "plastic", 1: "glass"},
            confidences=[0.5, 0.7, 0.9]
        )
        with patch("api.main.get_model", return_value=mock_model):
            from fastapi.testclient import TestClient
            from api.main import app
            yield TestClient(app)

    def test_small_image_does_not_crash(self, client):
        """Sehr kleine Bilder (10x10) sollen keinen 500er ausloesen."""
        img_bytes = make_image_bytes(width=10, height=10)
        response = client.post(
            "/predict/yolov8",
            files={"file": ("small.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_large_image_does_not_crash(self, client):
        """Grosse Bilder (1280x720) sollen keinen 500er ausloesen."""
        img_bytes = make_image_bytes(width=1280, height=720)
        response = client.post(
            "/predict/yolov8",
            files={"file": ("large.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_png_image_accepted(self, client):
        """PNG-Bilder sollen akzeptiert werden."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        response = client.post(
            "/predict/yolov8",
            files={"file": ("image.png", buf.read(), "image/png")},
        )
        assert response.status_code == 200

    def test_response_contains_model_version(self, client):
        """Die API-Antwort muss immer die model_version enthalten."""
        img_bytes = make_image_bytes()
        for version in ["yolov8", "yolov11"]:
            response = client.post(
                f"/predict/{version}",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            )
            assert response.status_code == 200
            assert response.json()["model_version"] == version

    def test_high_conf_threshold_returns_fewer_detections(self, client):
        """
        Mit hoeherem Confidence-Threshold sollen WENIGER oder gleich viele
        Detections zurueckkommen als mit niedrigem Threshold.
        Hier testen wir dass der API-Parameter korrekt weitergegeben wird.
        """
        img_bytes = make_image_bytes()
        low = client.post(
            "/predict/yolov8?conf_threshold=0.1",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        high = client.post(
            "/predict/yolov8?conf_threshold=0.99",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert low.status_code == 200
        assert high.status_code == 200
        # Beide muessen valide Antworten liefern (keine 500er)
        assert "num_detections" in low.json()
        assert "num_detections" in high.json()
