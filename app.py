"""
WasteVision Streamlit App – Frontend zur FastAPI.

Eingabe: Bild-Upload oder Kamera-Snapshot
Ausgabe: Detektionen vom Champion-Modell via FastAPI
"""

import io
import os
import requests
import streamlit as st
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="WasteVision 🗑️", layout="centered")
st.title("🗑️ WasteVision – Müll Erkennung")
st.caption("Powered by YOLOv8 & YOLOv11 | WasteVision MLOps Project")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Einstellungen")

conf_threshold = st.sidebar.slider(
    "Konfidenz-Schwellwert",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
)

with st.sidebar:
    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"✅ API verbunden\nModelle geladen: {health.get('models_loaded', [])}")
    except Exception:
        st.warning("⚠️ API nicht erreichbar.\nStarte die FastAPI mit:\n`uvicorn api.main:app`")

# ─── Champion Model Info ──────────────────────────────────────────────────────
try:
    champion_data = requests.get(f"{API_URL}/champion", timeout=2).json()
    model_version = champion_data["champion_version"]
    st.info(f"🏆 Aktives Modell: **{model_version}** (Pipeline-Champion) – {champion_data['num_classes']} Klassen")
except Exception:
    champion_data = {}
    model_version = "yolov8"
    st.info("🏆 Aktives Modell: **yolov8** (Standard-Fallback)")

with st.expander(f"ℹ️ Klassen für {model_version}"):
    try:
        cols = st.columns(3)
        for i, name in enumerate(champion_data.get("class_names", [])):
            cols[i % 3].write(f"• {name}")
    except Exception:
        st.info("Klasseninformation nicht verfügbar (API offline)")


# ─── Prediction Helper ────────────────────────────────────────────────────────
def run_prediction(image: Image.Image) -> None:
    """Schickt ein PIL-Bild an die FastAPI und zeigt die Ergebnisse an."""
    with st.spinner("Analysiere Bild..."):
        try:
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)

            response = requests.post(
                f"{API_URL}/predict",
                files={"file": ("image.jpg", buf, "image/jpeg")},
                params={"conf_threshold": conf_threshold},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            st.success(f"✅ {result['num_detections']} Objekt(e) erkannt")

            if result["detections"]:
                st.subheader("📊 Erkennungen")
                for det in result["detections"]:
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"**{det['class_name']}**")
                    col2.progress(det["confidence"], text=f"{det['confidence']:.0%}")
            else:
                st.info("Kein Müll über dem Schwellwert erkannt.")

        except requests.exceptions.ConnectionError:
            st.error("❌ API nicht erreichbar. Starte die FastAPI zuerst.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API Fehler: {e.response.text}")


# ─── Tabs: Upload / Kamera ────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📁 Bild hochladen", "📷 Kamera"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Bild auswählen (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Originalbild", use_column_width=True)
        if st.button("🔍 Erkennung starten", type="primary", key="btn_upload"):
            run_prediction(image)

with tab_camera:
    st.caption("Kamera-Snapshot wird direkt an das Champion-Modell gesendet.")
    camera_image = st.camera_input("Foto aufnehmen")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        run_prediction(image)
