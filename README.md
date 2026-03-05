# WasteVision MLOps Pipeline

Waste detection using YOLOv8 and YOLOv11 with a full MLOps infrastructure.
Built on top of Capstone 1 (transfer learning on YOLO models).

## Architecture

```
Streamlit Frontend
      |
      v
FastAPI (Port 8000)          <-- /metrics --> Prometheus (9090) --> Grafana (3000)
      |
      v
MLflow Model Registry (5001)
      |
      v
Prefect Evaluation Pipeline
      |
   +--+--+
   |     |
YOLOv8  YOLOv11
(7 Klassen) (26 Klassen)
```

## Models

| Modell | Klassen | Datensatz |
|--------|---------|-----------|
| YOLOv8 | 7 | cardboard, food_organics, glass, metal, miscellaneous_trash, paper, plastic |
| YOLOv11 | 26 | aerosol_cans, aluminum_cans, cardboard_boxes, clothing, coffee_grounds, ... |

## Tech Stack

| Thema | Tool |
|-------|------|
| API | FastAPI + Pydantic |
| ML Tracking | MLflow |
| Pipeline Orchestration | Prefect |
| Data Versioning | DVC |
| Testing | pytest (Unit, Integration, Behavioral) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |

## Quickstart

### Setup (lokal)

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Modelle platzieren

```bash
# .pt Dateien aus Capstone 1 in models/ kopieren
cp /pfad/zu/yolo8n_waste.pt  models/
cp /pfad/zu/yolo11n_waste.pt models/
```

### Stack starten (lokal ohne Docker)

```bash
# Terminal 1 – MLflow Server
./start_mlflow.sh local          # UI: http://localhost:5001

# Terminal 2 – FastAPI
uvicorn api.main:app --reload    # UI: http://localhost:8000/docs

# Terminal 3 – Streamlit (optional)
streamlit run app.py             # UI: http://localhost:8501
```

### Prefect Pipeline (Evaluation + Promotion)

```bash
# Prefect einmalig konfigurieren (falls noch nicht geschehen)
prefect config set PREFECT_SERVER_ALLOW_EPHEMERAL_MODE=true

# Pipeline starten (MLflow muss laufen)
python src/pipeline.py
```

Die Pipeline:
1. Validiert ob alle Modell- und Datendateien vorhanden sind
2. Evaluiert beide Modelle via `model.val()` und loggt Metriken in MLflow
3. Setzt das Modell mit höherem mAP50 als `champion`, das andere als `challenger`

### MLflow Modell manuell loggen

```bash
python src/train_mlflow.py \
    --model-path models/yolo8n_waste.pt \
    --version yolov8 \
    --data-yaml data/Yolo8/data_v8.yaml \
    --promote
```

### DVC Pipeline

```bash
dvc repro          # Führt alle Stages aus wenn Dependencies sich geändert haben
dvc params diff    # Zeigt Parameteränderungen
```

### Tests

```bash
pytest tests/ -v --ignore=tests/test_api
```

## Docker Compose (kompletter Stack)

```bash
docker compose up --build
```

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI | http://localhost:8000/docs | - |
| MLflow | http://localhost:5001 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / wastevision |

## Monitoring Metriken

Folgende Custom Metriken werden von der FastAPI an Prometheus geliefert:

| Metrik | Typ | Beschreibung |
|--------|-----|--------------|
| `wastevision_predict_requests_total` | Counter | Requests pro Modellversion |
| `wastevision_detections_total` | Counter | Erkannte Objekte nach Klasse |
| `wastevision_confidence_score` | Histogram | Verteilung der Confidence-Scores |
| `http_request_duration_seconds` | Histogram | API Latenz (automatisch) |

Das Grafana Dashboard (vorprovisioniert) zeigt Request-Rate, Latenz (p50/p95), Detections und Confidence-Verteilung.

## CI/CD

GitHub Actions führt bei jedem Push/PR auf `main` automatisch alle Tests aus.
Status: siehe Actions Tab im Repository.

## Projektstruktur

```
.
├── api/
│   └── main.py              # FastAPI mit Prometheus Metriken
├── src/
│   ├── predict.py            # YOLO Inferenz-Logik
│   ├── schemas.py            # Pydantic Schemas
│   ├── train_mlflow.py       # MLflow Logging + Model Registry
│   └── pipeline.py           # Prefect Evaluation Flow
├── tests/
│   ├── test_day1.py          # Unit + Integration Tests (FastAPI)
│   ├── test_day2.py          # MLflow Tests
│   ├── test_day3.py          # Pipeline + DVC Tests
│   └── test_behavioral.py    # Behavioral Tests (Modellverhalten)
├── data/
│   ├── Yolo8/data_v8.yaml   # Dataset Config (7 Klassen)
│   └── Yolo11/data_v11.yaml # Dataset Config (26 Klassen)
├── prometheus/prometheus.yml  # Scrape Config
├── grafana/provisioning/      # Auto-provisioned Dashboard
├── docker-compose.yml         # Gesamter Stack
├── Dockerfile                 # FastAPI Container
├── dvc.yaml                   # DVC Pipeline Stages
├── params.yaml                # Modell-Parameter
└── start_mlflow.sh            # MLflow Startup Script
```
