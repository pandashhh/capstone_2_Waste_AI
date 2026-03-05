"""
Tag 3 – Tests fuer die Prefect Pipeline und DVC-Konfiguration
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ─── Pipeline-Struktur Tests ─────────────────────────────────────────────────
class TestPipelineConfig:
    def test_dvc_yaml_exists(self):
        assert Path("dvc.yaml").exists(), "dvc.yaml fehlt"

    def test_dvc_yaml_has_required_stages(self):
        with open("dvc.yaml") as f:
            config = yaml.safe_load(f)
        stages = config.get("stages", {})
        assert "evaluate_yolov8" in stages
        assert "evaluate_yolov11" in stages

    def test_dvc_yaml_stages_have_deps(self):
        with open("dvc.yaml") as f:
            config = yaml.safe_load(f)
        for stage_name, stage in config["stages"].items():
            assert "deps" in stage, f"Stage '{stage_name}' hat keine deps"
            assert len(stage["deps"]) > 0

    def test_params_yaml_exists(self):
        assert Path("params.yaml").exists(), "params.yaml fehlt"

    def test_params_yaml_has_model_configs(self):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        assert "yolov8" in params
        assert "yolov11" in params
        for version in ["yolov8", "yolov11"]:
            assert "conf_threshold" in params[version]
            assert "imgsz" in params[version]

    def test_pipeline_file_exists(self):
        assert Path("src/pipeline.py").exists(), "src/pipeline.py fehlt"


# ─── Pipeline-Logik Tests (mit Mocks) ────────────────────────────────────────
class TestValidateData:
    def test_validate_data_raises_on_missing_model(self, tmp_path):
        """validate_data soll einen Fehler werfen wenn Modelldatei fehlt."""
        from src.pipeline import MODELS
        with patch.dict(MODELS, {
            "yolov8": {
                "path": str(tmp_path / "missing.pt"),
                "data_yaml": str(tmp_path / "missing.yaml"),
                "registry_name": "wastevision-yolov8",
            }
        }, clear=True):
            with patch("src.pipeline.get_run_logger", return_value=MagicMock()):
                from src.pipeline import validate_data
                with pytest.raises(FileNotFoundError):
                    validate_data.fn()

    def test_validate_data_passes_when_files_exist(self, tmp_path):
        """validate_data soll ohne Fehler durchlaufen wenn alle Dateien existieren."""
        model_file = tmp_path / "model.pt"
        yaml_file = tmp_path / "data.yaml"
        model_file.touch()
        yaml_file.touch()

        from src.pipeline import MODELS
        with patch.dict(MODELS, {
            "yolov8": {
                "path": str(model_file),
                "data_yaml": str(yaml_file),
                "registry_name": "wastevision-yolov8",
            }
        }, clear=True):
            with patch("src.pipeline.get_run_logger", return_value=MagicMock()):
                from src.pipeline import validate_data
                validate_data.fn()  # soll keinen Fehler werfen


class TestPromoteBestModel:
    def test_promotes_model_with_higher_map50(self):
        """Das Modell mit hoeherem mAP50 soll Champion werden."""
        mock_client = MagicMock()

        mock_run_v8 = MagicMock()
        mock_run_v8.data.metrics = {"mAP50": 0.75}

        mock_run_v11 = MagicMock()
        mock_run_v11.data.metrics = {"mAP50": 0.82}

        mock_client.get_run.side_effect = lambda run_id: (
            mock_run_v8 if run_id == "run_v8" else mock_run_v11
        )

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_client.search_model_versions.return_value = [mock_version]

        with patch("src.pipeline.MlflowClient", return_value=mock_client), \
             patch("src.pipeline.mlflow.set_tracking_uri"), \
             patch("src.pipeline.get_run_logger", return_value=MagicMock()):

            from src.pipeline import promote_best_model
            result = promote_best_model.fn({"yolov8": "run_v8", "yolov11": "run_v11"})

        assert result == "yolov11", f"Erwartet yolov11, bekommen: {result}"

    def test_champion_alias_is_set(self):
        """set_registered_model_alias muss mit 'champion' aufgerufen werden."""
        mock_client = MagicMock()

        for run_id in ["run_v8", "run_v11"]:
            mock_run = MagicMock()
            mock_run.data.metrics = {"mAP50": 0.8 if run_id == "run_v8" else 0.7}
            mock_client.get_run.side_effect = lambda rid: (
                MagicMock(**{"data.metrics": {"mAP50": 0.8}}) if rid == "run_v8"
                else MagicMock(**{"data.metrics": {"mAP50": 0.7}})
            )

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_client.search_model_versions.return_value = [mock_version]

        with patch("src.pipeline.MlflowClient", return_value=mock_client), \
             patch("src.pipeline.mlflow.set_tracking_uri"), \
             patch("src.pipeline.get_run_logger", return_value=MagicMock()):

            from src.pipeline import promote_best_model
            promote_best_model.fn({"yolov8": "run_v8", "yolov11": "run_v11"})

        alias_calls = [
            call.args[1]
            for call in mock_client.set_registered_model_alias.call_args_list
        ]
        assert "champion" in alias_calls
        assert "challenger" in alias_calls
