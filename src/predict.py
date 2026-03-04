
"""
Inference logic for Waste Detection YOLO models.
Supports YOLOv8 (7 classes) and YOLOv11 (26 classes).
"""

import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path


# Class labels per model version
YOLO8_CLASSES = [
    "cardboard", "food_organics", "glass", "metal",
    "miscellaneous_trash", "paper", "plastic"
]

YOLO11_CLASSES = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans",
    "cardboard_boxes", "cardboard_packaging", "clothing",
    "coffee_grounds", "disposable_plastic_cutlery", "eggshells",
    "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper",
    "paper_cups", "plastic_detergent_bottles", "plastic_food_containers",
    "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws",
    "plastic_trash_bags", "plastic_water_bottles", "shoes", "styrofoam_cups"
]


def load_model(model_path: str) -> YOLO:
    """Load a YOLO model from the given path."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return YOLO(str(path))


def predict_image(model: YOLO, image: np.ndarray, conf_threshold: float = 0.45) -> dict:
    """
    Run inference on a single image array (RGB, HWC).

    Returns a dict with:
        - detections: list of {class_id, class_name, confidence, bbox}
        - annotated_image: np.ndarray with drawn bounding boxes
        - num_detections: int
    """
    results = model(image, conf=conf_threshold)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bbox": [round(v, 2) for v in bbox],
        })

    annotated_image = result.plot()

    return {
        "detections": detections,
        "annotated_image": annotated_image,
        "num_detections": len(detections),
    }


def predict_from_pil(model: YOLO, pil_image: Image.Image, conf_threshold: float = 0.45) -> dict:
    """Convenience wrapper: accepts a PIL Image."""
    img_array = np.array(pil_image.convert("RGB"))
    return predict_image(model, img_array, conf_threshold)
