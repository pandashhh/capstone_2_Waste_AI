"""
WasteVision Streamlit App – jetzt als Frontend zur FastAPI.

Die App schickt Bilder an die FastAPI und zeigt die Ergebnisse an.
Für lokale Entwicklung ohne API kann der direkte Modus verwendet werden.
"""

import io
import os
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ─── Config ──────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="WasteVision 🗑️", layout="centered")
st.title("🗑️ WasteVision – Müll Erkennung")
st.caption("Powered by YOLOv8 & YOLOv11 | WasteVision MLOps Project")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Einstellungen")

model_version = st.sidebar.radio(
    "Modell auswählen",
    options=["yolov8", "yolov11"],
    captions=["7 Klassen (grob)", "26 Klassen (detailliert)"],
)

conf_threshold = st.sidebar.slider(
    "Konfidenz-Schwellwert",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
)

# ─── API Health Check ─────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"✅ API verbunden\nModelle geladen: {health.get('models_loaded', [])}")
    except Exception:
        st.warning("⚠️ API nicht erreichbar.\nStarte die FastAPI mit:\n`uvicorn api.main:app`")

# ─── Model Info ──────────────────────────────────────────────────────────────
with st.expander(f"ℹ️ Klassen für {model_version}"):
    try:
        info = requests.get(f"{API_URL}/models/{model_version}", timeout=2).json()
        cols = st.columns(3)
        for i, name in enumerate(info["class_names"]):
            cols[i % 3].write(f"• {name}")
    except Exception:
        st.info("Klasseninformation nicht verfügbar (API offline)")

# ─── Image Upload ─────────────────────────────────────────────────────────────
st.subheader("📷 Bild hochladen")
uploaded_file = st.file_uploader(
    "Bild auswählen (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Originalbild", use_column_width=True)

    if st.button("🔍 Erkennung starten", type="primary"):
        with st.spinner("Analysiere Bild..."):
            try:
                # Send to FastAPI
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                buf.seek(0)

                response = requests.post(
                    f"{API_URL}/predict/{model_version}",
                    files={"file": ("image.jpg", buf, "image/jpeg")},
                    params={"conf_threshold": conf_threshold},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

                # ─── Results ─────────────────────────────────────────────
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