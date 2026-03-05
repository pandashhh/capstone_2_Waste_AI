"""
Tests für Tag 2 – MLflow Integration
Alles gemockt, kein echter MLflow Server nötig.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, call
from pathlib import Path


# ─── Helper ──────────────────────────────────────────────────────────────────

def make_mock_yolo_metrics():
    """Simuliert das Rückgabeobjekt von model.val()"""
    metrics = MagicMock()
    metrics.box.map50 = 0.82
    metrics.box.map   = 0.61
    metrics.box.mp    = 0.78
    metrics.box.mr    = 0.74
    return metrics


# ─── Unit Tests: extract_metrics ─────────────────────────────────────────────

class TestExtractMetrics:

    def test_returns_all_four_keys(self):
        from src.train_mlflow import extract_metrics
        mock_metrics = make_mock_yolo_metrics()
        result = extract_metrics(mock_metrics)
        assert set(result.keys()) == {"mAP50", "mAP50_95", "precision", "recall"}

    def test_values_are_rounded_floats(self):
        from src.train_mlflow import extract_metrics
        mock_metrics = make_mock_yolo_metrics()
        result = extract_metrics(mock_metrics)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} should be float"
            assert 0.0 <= v <= 1.0, f"{k} should be between 0 and 1"

    def test_correct_values(self):
        from src.train_mlflow import extract_metrics
        mock_metrics = make_mock_yolo_metrics()
        result = extract_metrics(mock_metrics)
        assert result["mAP50"]    == 0.82
        assert result["mAP50_95"] == 0.61
        assert result["precision"] == 0.78
        assert result["recall"]    == 0.74


# ─── Unit Tests: log_model_to_mlflow (gemockt) ───────────────────────────────

class TestLogModelToMlflow:

    @patch("src.train_mlflow.mlflow")
    @patch("src.train_mlflow.YOLO")
    def test_raises_if_model_file_missing(self, mock_yolo, mock_mlflow):
        from src.train_mlflow import log_model_to_mlflow
        with pytest.raises(FileNotFoundError):
            log_model_to_mlflow(
                model_path="nonexistent/model.pt",
                version="yolov8",
                data_yaml=None,
                tracking_uri="http://localhost:5000",
            )

    @patch("src.train_mlflow.mlflow")
    @patch("src.train_mlflow.YOLO")
    @patch("pathlib.Path.exists", return_value=True)
    def test_sets_experiment_name(self, mock_exists, mock_yolo, mock_mlflow):
        from src.train_mlflow import log_model_to_mlflow, EXPERIMENT_NAME

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "test-run-id-123"
        mock_mlflow.start_run.return_value = mock_run

        log_model_to_mlflow("models/fake.pt", "yolov8", None, "http://localhost:5000")

        mock_mlflow.set_experiment.assert_called_once_with(EXPERIMENT_NAME)

    @patch("src.train_mlflow.mlflow")
    @patch("src.train_mlflow.YOLO")
    @patch("pathlib.Path.exists", return_value=True)
    def test_logs_params_with_correct_keys(self, mock_exists, mock_yolo, mock_mlflow):
        from src.train_mlflow import log_model_to_mlflow

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "abc"
        mock_mlflow.start_run.return_value = mock_run

        log_model_to_mlflow("models/fake.pt", "yolov8", None, "http://localhost:5000")

        call_args = mock_mlflow.log_params.call_args[0][0]
        assert "model_path" in call_args
        assert "num_classes" in call_args
        assert call_args["num_classes"] == 7  # yolov8 hat 7 Klassen

    @patch("src.train_mlflow.mlflow")
    @patch("src.train_mlflow.YOLO")
    @patch("pathlib.Path.exists", return_value=True)
    def test_yolov11_has_26_classes(self, mock_exists, mock_yolo, mock_mlflow):
        from src.train_mlflow import log_model_to_mlflow

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "abc"
        mock_mlflow.start_run.return_value = mock_run

        log_model_to_mlflow("models/fake.pt", "yolov11", None, "http://localhost:5000")

        call_args = mock_mlflow.log_params.call_args[0][0]
        assert call_args["num_classes"] == 26


# ─── Unit Tests: promote_to_production ───────────────────────────────────────

class TestPromoteToProduction:

    @patch("src.train_mlflow.mlflow")
    def test_sets_champion_alias(self, mock_mlflow):
        from src.train_mlflow import promote_to_production

        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        version = MagicMock()
        version.version = "3"
        mock_client.search_model_versions.return_value = [version]

        promote_to_production("wastevision-yolov8", "http://localhost:5000")

        mock_client.set_registered_model_alias.assert_called_once_with(
            name="wastevision-yolov8",
            alias="champion",
            version="3",
        )

    @patch("src.train_mlflow.mlflow")
    def test_no_versions_does_not_crash(self, mock_mlflow):
        from src.train_mlflow import promote_to_production

        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_client.search_model_versions.return_value = []

        # Sollte keinen Fehler werfen
        promote_to_production("wastevision-yolov8", "http://localhost:5000")
        mock_client.set_registered_model_alias.assert_not_called()

    @patch("src.train_mlflow.mlflow")
    def test_picks_latest_version(self, mock_mlflow):
        from src.train_mlflow import promote_to_production

        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        v1, v2, v3 = MagicMock(), MagicMock(), MagicMock()
        v1.version, v2.version, v3.version = "1", "3", "2"
        mock_client.search_model_versions.return_value = [v1, v2, v3]

        promote_to_production("wastevision-yolov8", "http://localhost:5000")

        # Version 3 ist die neueste
        mock_client.set_registered_model_alias.assert_called_once_with(
            name="wastevision-yolov8",
            alias="champion",
            version="3",
        )
