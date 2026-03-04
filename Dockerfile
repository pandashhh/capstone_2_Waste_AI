# ── WasteVision API ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

