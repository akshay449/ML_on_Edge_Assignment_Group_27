"""
=========================================================
LogiBridge - Real-Time Edge Inference Service
Module 5 - Task E1
=========================================================
Listens to live MQTT raw telemetry, buffers samples into the same
5-sample-MA / 30s-window / 10s-step feature vectors that
preprocessing.py builds offline (imported from the same module, so
training and serving can never drift apart), normalises with the
frozen training_stats.json baseline, runs TFLite inference, prints the
prediction, and publishes the result to the inference MQTT topic.
=========================================================
"""

import glob
import json
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "data_pipeline")
if DATA_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, DATA_PIPELINE_DIR)

from preprocessing import FEATURE_COLUMNS, extract_window_features, MA_WINDOW, WINDOW_SECONDS, STEP_SECONDS

# Configuration / Environment Variables
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
TRUCK_ID = os.getenv("TRUCK_ID", "truck_001")
SENSOR_TOPIC = f"logibridge/{TRUCK_ID}/sensors"
INFERENCE_TOPIC = f"logibridge/trucks/{TRUCK_ID}/inference"


def find_model_path():
    """Locates any valid TFLite model file inside container or local directories.
    Priority: pruned+quantised (M3) > PTQ-only (M2) > FP32 baseline (M1)."""
    env_path = os.getenv("MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    possible_paths = [
        "/app/models/logibridge_pruned_int8.tflite",
        "/app/models/logibridge_int8.tflite",
        "/app/models/cold_chain_model.tflite",
        os.path.join(PROJECT_ROOT, "models", "logibridge_pruned_int8.tflite"),
        os.path.join(PROJECT_ROOT, "models", "logibridge_int8.tflite"),
        os.path.join(PROJECT_ROOT, "models", "cold_chain_model.tflite"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    search_dirs = ["/app/models", os.path.join(PROJECT_ROOT, "models"), "models", "/app"]
    for s_dir in search_dirs:
        tflite_files = glob.glob(os.path.join(s_dir, "*.tflite"))
        if tflite_files:
            return tflite_files[0]

    raise FileNotFoundError("Could not locate any valid TFLite model file in /app/models/ or models/")


def load_training_stats():
    """Loads the FROZEN Normal-class baseline stats (see preprocessing.py)."""
    possible_stats_paths = [
        "/app/data_pipeline/training_stats.json",
        os.path.join(DATA_PIPELINE_DIR, "training_stats.json"),
        "training_stats.json",
    ]
    for path in possible_stats_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    # Fallback default statistics if the frozen stats file is missing.
    print("[WARNING] training_stats.json not found. Using default normalization stats "
          "-- predictions will be unreliable until generate_dataset.py has run.")
    return {
        "temp_mean": {"mean": 4.0, "std": 1.0},
        "temp_std": {"mean": 0.3, "std": 0.2},
        "temp_slope": {"mean": 0.0, "std": 0.5},
        "vib_rms": {"mean": 0.45, "std": 0.1},
        "vib_peak": {"mean": 0.55, "std": 0.15},
        "vib_kurtosis": {"mean": 0.0, "std": 1.0},
        "door_open_ratio": {"mean": 0.1, "std": 0.1},
    }


class SlidingWindower:
    """Maintains a rolling buffer of raw samples and emits one feature
    vector every STEP_SECONDS new samples, once WINDOW_SECONDS worth of
    history is available -- mirroring preprocessing.py's offline windowing."""

    def __init__(self, window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS, ma_window=MA_WINDOW):
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.ma_window = ma_window
        self.temp_buf = deque(maxlen=window_seconds)
        self.vib_buf = deque(maxlen=window_seconds)
        self.door_buf = deque(maxlen=window_seconds)
        self._since_last_window = 0

    def add_sample(self, temperature, vibration_rms, door_open):
        self.temp_buf.append(temperature)
        self.vib_buf.append(vibration_rms)
        self.door_buf.append(door_open)
        self._since_last_window += 1

        if len(self.temp_buf) < self.window_seconds:
            return None
        if self._since_last_window < self.step_seconds:
            return None

        self._since_last_window = 0

        temp_ma = pd.Series(self.temp_buf).rolling(window=self.ma_window, min_periods=1).mean().values
        vib_ma = pd.Series(self.vib_buf).rolling(window=self.ma_window, min_periods=1).mean().values
        door_arr = np.array(self.door_buf)

        return extract_window_features(temp_ma, vib_ma, door_arr)


def quantize_input(features_array, input_details):
    dtype = input_details["dtype"]
    if dtype in (np.int8, np.uint8):
        # Modern TFLite converter output populates 'quantization_parameters'
        # (scales/zero_points arrays), not the legacy 'quantization' tuple --
        # see benchmark.py's _get_scale_zero_point for the full explanation.
        qparams = input_details.get("quantization_parameters")
        if qparams and len(qparams.get("scales", [])) > 0:
            scale = float(qparams["scales"][0])
            zero_point = int(qparams["zero_points"][0])
        else:
            scale, zero_point = input_details.get("quantization", (0.0, 0))
            if not scale:
                raise ValueError(
                    "No valid quantization scale found for the model's input tensor -- "
                    "check that the deployed .tflite was produced by convert_ptq.py / "
                    "prune_quantise.py with proper calibration."
                )
        q = features_array / scale + zero_point
        return np.clip(np.round(q), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return features_array.astype(np.float32)


def dequantize_output(y_raw, output_details):
    dtype = output_details["dtype"]
    if dtype in (np.int8, np.uint8):
        qparams = output_details.get("quantization_parameters")
        if qparams and len(qparams.get("scales", [])) > 0:
            scale = float(qparams["scales"][0])
            zero_point = int(qparams["zero_points"][0])
        else:
            scale, zero_point = output_details.get("quantization", (0.0, 0))
            if not scale:
                raise ValueError("No valid quantization scale found for the model's output tensor.")
        return (y_raw.astype(np.float32) - zero_point) * scale
    return y_raw.astype(np.float32)


# Initialize TFLite Interpreter
model_path = find_model_path()
print(f"Loading TFLite model from: {model_path}")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

is_quantized = input_details["dtype"] in (np.int8, np.uint8)
print(f"Model successfully loaded | Data Type: {input_details['dtype']} | Quantized: {is_quantized}")

# Load frozen normalisation parameters and set up the live windower
stats = load_training_stats()
windower = SlidingWindower()
window_index = 0

# Label Mapping
CLASS_MAP = {0: "NORMAL", 1: "MODERATE RISK", 2: "CRITICAL ALERT (Spoilage Risk)"}


def preprocess_sample(telemetry):
    """Feeds one raw MQTT sample into the sliding windower. Returns a
    normalised, model-ready feature array, or None if not enough history
    has accumulated yet for a full window."""
    temp = float(telemetry.get("temperature", 4.0))
    vib = float(telemetry.get("vibration_rms", 0.5))

    door_raw = str(telemetry.get("door_event", "CLOSE")).upper()
    door_open = 1.0 if door_raw == "OPEN" or telemetry.get("door_open", 0) == 1 else 0.0

    features = windower.add_sample(temp, vib, door_open)
    if features is None:
        return None

    normed = []
    for col in FEATURE_COLUMNS:
        mean_val = stats[col]["mean"]
        std_val = stats[col]["std"] or 1e-6
        normed.append((features[col] - mean_val) / std_val)

    feature_array = np.array([normed], dtype=np.float32)
    return quantize_input(feature_array, input_details)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Successfully connected to MQTT Broker '{MQTT_BROKER}' on topic '{SENSOR_TOPIC}'")
        client.subscribe(SENSOR_TOPIC)
    else:
        print(f"Failed to connect to MQTT broker, return code: {rc}")


def on_message(client, userdata, msg):
    global window_index
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        input_data = preprocess_sample(payload)

        if input_data is None:
            # Still buffering -- not enough samples yet for a full 30s window.
            return

        # Run TFLite Inference
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"])
        # Dequantize before computing confidence: argmax is unaffected by
        # dequantization (it's a monotonic per-tensor transform), but the
        # raw int8 output is NOT a 0-1 probability -- reporting it directly
        # as "confidence" (previous behaviour) would publish values like
        # -128..127 instead, breaking drift_monitor.py's PSI binning, which
        # expects confidence in [0,1].
        output_data = dequantize_output(raw_output, output_details) if is_quantized else raw_output

        predicted_class = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        risk_label = CLASS_MAP.get(predicted_class, "UNKNOWN")
        window_index += 1

        print(
            f"[{payload.get('timestamp', 'N/A')}] Truck: {payload.get('truck_id', TRUCK_ID)} | "
            f"Window #{window_index} | Temp: {payload.get('temperature'):.2f}C | "
            f"Vib: {payload.get('vibration_rms'):.2f}g | "
            f"Prediction: [{predicted_class}] {risk_label} (confidence={confidence:.4f})"
        )

        result_payload = json.dumps({
            "truck_id": payload.get("truck_id", TRUCK_ID),
            "window_index": window_index,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
        })
        client.publish(INFERENCE_TOPIC, result_payload, qos=1)

    except Exception as e:
        print(f"[ERROR] Failed to process telemetry message: {e}")


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopping Edge Inference Service.")
        client.disconnect()
    except Exception as e:
        print(f"Connection error: {e}")


if __name__ == "__main__":
    main()
