"""
Tag 3 - Prefect Evaluation Pipeline fuer WasteVision MLOps

Flow:
    validate_data -> evaluate_yolov8 --+
                                       +--> promote_best_model
                 -> evaluate_yolov11 --+

Verwendung:
    # MLflow Server muss laufen (./start_mlflow.sh local)
    python src/pipeline.py
"""

import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task, get_run_logger

sys.path.insert(0, str(Path(__file__).parent))
from train_mlflow import log_model_to_mlflow

# --- Konfiguration -----------------------------------------------------------
TRACKING_URI = "http://localhost:5001"

MODELS = {
    "yolov8": {
        "path": "models/yolo8n_waste.pt",
        "data_yaml": "data/Yolo8/data_v8.yaml",
        "registry_name": "wastevision-yolov8",
    },
    "yolov11": {
        "path": "models/yolo11n_waste.pt",
        "data_yaml": "data/Yolo11/data_v11.yaml",
        "registry_name": "wastevision-yolov11",
    },
}


# --- Tasks -------------------------------------------------------------------
@task(name="validate-data", retries=0)
def validate_data() -> None:
    """Prueft ob alle Modelldateien und data.yaml Dateien vorhanden sind."""
    logger = get_run_logger()
    missing = []

    for version, config in MODELS.items():
        model_ok = Path(config["path"]).exists()
        yaml_ok = Path(config["data_yaml"]).exists()
        status = "OK" if (model_ok and yaml_ok) else "FEHLT"
        logger.info(f"[{status}] {version}: model={model_ok}, yaml={yaml_ok}")

        if not model_ok:
            missing.append(config["path"])
        if not yaml_ok:
            missing.append(config["data_yaml"])

    if missing:
        raise FileNotFoundError(f"Fehlende Dateien: {missing}")

    logger.info("Alle Dateien vorhanden.")


@task(name="evaluate-model", retries=1)
def evaluate_model(version: str) -> str:
    """Fuehrt model.val() aus, loggt Metriken in MLflow und gibt die Run-ID zurueck."""
    logger = get_run_logger()
    config = MODELS[version]

    logger.info(f"Starte Evaluation fuer {version}...")
    run_id = log_model_to_mlflow(
        model_path=config["path"],
        version=version,
        data_yaml=config["data_yaml"],
        tracking_uri=TRACKING_URI,
    )
    logger.info(f"{version} abgeschlossen. Run ID: {run_id}")
    return run_id


@task(name="promote-best-model")
def promote_best_model(run_ids: dict) -> str:
    """
    Vergleicht mAP50 der beiden Runs und setzt das bessere Modell als 'champion'.
    Das schlechtere Modell bekommt den Alias 'challenger'.
    """
    logger = get_run_logger()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    scores = {}
    for version, run_id in run_ids.items():
        run = client.get_run(run_id)
        map50 = run.data.metrics.get("mAP50", 0.0)
        scores[version] = map50
        logger.info(f"{version}: mAP50={map50:.4f}")

    best = max(scores, key=scores.get)
    challenger = [v for v in scores if v != best][0]

    for version, alias in [(best, "champion"), (challenger, "challenger")]:
        model_name = MODELS[version]["registry_name"]
        versions = client.search_model_versions(f"name='{model_name}'")
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.set_registered_model_alias(model_name, alias, latest.version)
        logger.info(
            f"{model_name} v{latest.version} -> alias='{alias}' "
            f"(mAP50={scores[version]:.4f})"
        )

    logger.info(f"Champion: {best} (mAP50={scores[best]:.4f})")
    return best


# --- Flow --------------------------------------------------------------------
@flow(name="wastevision-evaluation-pipeline", log_prints=True)
def evaluation_pipeline() -> str:
    """
    Hauptflow: Validierung -> Evaluation beider Modelle -> Promotion des Besten.
    """
    validate_data()

    run_ids = {}
    for version in MODELS:
        run_ids[version] = evaluate_model(version)

    champion = promote_best_model(run_ids)

    print(f"\nPipeline abgeschlossen. Champion: {champion}")
    print(f"MLflow UI: {TRACKING_URI}")
    return champion


if __name__ == "__main__":
    evaluation_pipeline()
