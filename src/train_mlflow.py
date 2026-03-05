"""
Tag 2 – MLflow Experiment Tracking für WasteVision

Was dieser Script macht:
1. Lädt ein bereits trainiertes YOLO-Modell (.pt Datei)
2. Führt model.val() aus um Metriken zu berechnen
3. Loggt alle Metriken + das Modell in MLflow
4. Registriert das beste Modell im MLflow Model Registry

Verwendung:
    # YOLOv8 loggen
    python src/train_mlflow.py --model-path models/yolo8n_waste.pt --version yolov8

    # YOLOv11 loggen  
    python src/train_mlflow.py --model-path models/yolo11n_waste.pt --version yolov11

    # Mit eigenem Tracking URI (z.B. lokaler MLflow Server)
    python src/train_mlflow.py --model-path models/yolo8n_waste.pt --version yolov8 \
        --tracking-uri http://localhost:5001
"""

import argparse
import os
import json
from pathlib import Path

import mlflow
import mlflow.pyfunc
from ultralytics import YOLO

from predict import YOLO8_CLASSES, YOLO11_CLASSES


# ─── Konstanten ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "wastevision-waste-detection"

CLASS_NAMES = {
    "yolov8": YOLO8_CLASSES,
    "yolov11": YOLO11_CLASSES,
}


# ─── MLflow Custom Model Wrapper ─────────────────────────────────────────────
class YOLOWrapper(mlflow.pyfunc.PythonModel):
    """
    Wraps YOLO so MLflow kann es als pyfunc-Modell speichern und laden.
    Input:  dict mit {"image_path": str}
    Output: dict mit detections
    """

    def load_context(self, context):
        model_path = context.artifacts["yolo_model"]
        self.model = YOLO(model_path)

    def predict(self, context, model_input):
        import numpy as np
        from PIL import Image

        image_path = model_input["image_path"].iloc[0]
        img = np.array(Image.open(image_path).convert("RGB"))
        results = self.model(img, conf=0.45)
        result = results[0]

        detections = []
        for box in result.boxes:
            detections.append({
                "class_id": int(box.cls.item()),
                "class_name": result.names[int(box.cls.item())],
                "confidence": round(float(box.conf.item()), 4),
                "bbox": [round(v, 2) for v in box.xyxy[0].tolist()],
            })
        return {"detections": detections, "num_detections": len(detections)}


# ─── Metriken aus YOLO val() extrahieren ─────────────────────────────────────
def extract_metrics(metrics) -> dict:
    """Extrahiert die wichtigsten Metriken aus dem YOLO val() Ergebnis."""
    return {
        "mAP50":       round(float(metrics.box.map50), 4),
        "mAP50_95":    round(float(metrics.box.map),   4),
        "precision":   round(float(metrics.box.mp),    4),
        "recall":      round(float(metrics.box.mr),    4),
    }


# ─── Haupt-Logging Funktion ───────────────────────────────────────────────────
def log_model_to_mlflow(
    model_path: str,
    version: str,
    data_yaml: str | None,
    tracking_uri: str,
) -> str:
    """
    Lädt das Modell, berechnet Metriken und loggt alles in MLflow.
    Gibt die run_id zurück.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

    print(f"\n🔄 Lade Modell: {model_path}")
    yolo_model = YOLO(str(model_path))

    with mlflow.start_run(run_name=f"{version}_evaluation") as run:
        run_id = run.info.run_id
        print(f"📊 MLflow Run ID: {run_id}")

        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_version": version,
            "num_classes":   len(CLASS_NAMES[version]),
            "framework":     "ultralytics",
            "project":       "wastevision",
        })

        # ── Parameter loggen ──────────────────────────────────────────────────
        mlflow.log_params({
            "model_path":    str(model_path),
            "class_names":   json.dumps(CLASS_NAMES[version]),
            "num_classes":   len(CLASS_NAMES[version]),
            "conf_threshold": 0.45,
            "imgsz":         640,
        })

        # ── Validierung + Metriken loggen ─────────────────────────────────────
        if data_yaml and Path(data_yaml).exists():
            print("📐 Führe model.val() aus...")
            metrics = yolo_model.val(data=data_yaml, imgsz=640, verbose=False)
            extracted = extract_metrics(metrics)
            mlflow.log_metrics(extracted)
            print(f"✅ Metriken: {extracted}")
        else:
            print("⚠️  Kein data.yaml angegeben – Metriken werden übersprungen.")
            print("    Füge --data-yaml <pfad/data.yaml> hinzu um Metriken zu loggen.")

        # ── Modell in MLflow speichern ────────────────────────────────────────
        print("💾 Speichere Modell in MLflow...")
        artifacts = {"yolo_model": str(model_path)}

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=YOLOWrapper(),
            artifacts=artifacts,
            registered_model_name=f"wastevision-{version}",
            pip_requirements=[
                f"ultralytics==8.3.0",
                "numpy==1.26.4",
                "Pillow==10.4.0",
            ],
        )
        print(f"✅ Modell registriert als: wastevision-{version}")

    return run_id


# ─── Model Registry: beste Version auf 'Production' setzen ───────────────────
def promote_to_production(model_name: str, tracking_uri: str):
    """
    Setzt die neueste Version eines registrierten Modells auf 'Champion'.
    (MLflow 2.x nutzt Aliases statt Stages)
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"⚠️  Kein Modell '{model_name}' gefunden.")
        return

    # Neueste Version nehmen
    latest = sorted(versions, key=lambda v: int(v.version))[-1]

    # MLflow 2.x: Alias setzen statt Stage
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=latest.version,
    )
    print(f"🏆 {model_name} v{latest.version} → alias='champion'")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Logge ein YOLO-Modell in MLflow"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Pfad zur .pt Datei, z.B. models/yolo8n_waste.pt"
    )
    parser.add_argument(
        "--version", required=True, choices=["yolov8", "yolov11"],
        help="Modellversion"
    )
    parser.add_argument(
        "--data-yaml", default=None,
        help="Pfad zur data.yaml für model.val() (optional)"
    )
    parser.add_argument(
        "--tracking-uri", default="http://localhost:5001",
        help="MLflow Tracking URI (default: http://localhost:5001)"
    )
    parser.add_argument(
        "--promote", action="store_true",
        help="Neueste Version nach dem Loggen auf 'champion' setzen"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_id = log_model_to_mlflow(
        model_path=args.model_path,
        version=args.version,
        data_yaml=args.data_yaml,
        tracking_uri=args.tracking_uri,
    )

    if args.promote:
        promote_to_production(
            model_name=f"wastevision-{args.version}",
            tracking_uri=args.tracking_uri,
        )

    print(f"\n✅ Fertig! Run ID: {run_id}")
    print(f"   MLflow UI: {args.tracking_uri.replace('/api', '')} → Experiments → {EXPERIMENT_NAME}")
