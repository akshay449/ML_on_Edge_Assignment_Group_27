# Use lightweight python 3.11 base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- Layer 1: dependencies (changes rarest -> cached almost always) ---
COPY inference/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# --- Layer 2: pipeline / service code (changes occasionally) ---
# Only preprocessing.py is needed at runtime from data_pipeline/ (it's
# imported by inference_service.py for FEATURE_COLUMNS / windowing math).
COPY data_pipeline/preprocessing.py /app/data_pipeline/preprocessing.py
COPY inference/ /app/inference/

# --- Layer 3: model + frozen stats (changes every OTA update) ---
# These two COPY instructions are deliberately LAST. On an OTA model swap,
# only this layer (and nothing before it) gets invalidated and rebuilt --
# this is the layer-cache behaviour Task D2 asks you to demonstrate:
#   1. docker build ...          (full build, all layers)
#   2. replace models/logibridge_pruned_int8.tflite with a new version
#   3. docker build ...          (rebuild -- watch only these two COPY
#                                  steps + nothing above them re-run)
COPY models/ /app/models/
COPY data_pipeline/training_stats.json /app/data_pipeline/training_stats.json

# Default Environment Variables
ENV MODEL_PATH="/app/models/logibridge_pruned_int8.tflite"
ENV MQTT_BROKER="localhost"
ENV MQTT_PORT=1883
ENV TRUCK_ID="truck_001"

# Run inference service
CMD ["python", "inference/inference_service.py"]
