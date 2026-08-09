"""
=========================================================
LogiEdge - Cold Chain Sensor Simulator
Module 3 - Task C1
=========================================================
Publishes 3 streams:
- Temperature : 1 Hz (Setpoint 4.0°C)
- Vibration   : 0.5 Hz (Compressor RMS)
- Door Events : Discrete ('OPEN' / 'CLOSE')
Publishes to Local MQTT Broker: logibridge/truck_001/sensors
=========================================================
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
import paho.mqtt.client as mqtt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "sensor_data.csv")
CSV_HEADERS = ["timestamp", "truck_id", "reading", "temperature", "vibration_rms", "door_event", "anomaly"]

BROKER = "localhost"
PORT = 1883
TOPIC = "logibridge/truck_001/sensors"

parser = argparse.ArgumentParser(description="LogiEdge Cold Chain Sensor Simulator")
parser.add_argument("--anomaly", choices=["none", "temp_drift", "vibration", "combined"], default="none")
parser.add_argument("--minutes", type=int, default=1, help="Simulation duration in minutes")
parser.add_argument("--fast", action="store_true",
                     help="Skip the 1s-per-reading real-time pacing. Safe ONLY for offline dataset "
                          "generation (generate_dataset.py) -- the random sampling and drift formulas "
                          "don't depend on wall-clock time at all, so output is statistically identical, "
                          "just produced instantly instead of over --minutes of real time. Do NOT use "
                          "--fast for the live MQTT pipeline demo, where real 1Hz pacing is the point.")
args = parser.parse_args()

TOTAL_SECONDS = args.minutes * 60

def initialize_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)

def save_to_csv(sensor_data):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            sensor_data["timestamp"],
            sensor_data["truck_id"],
            sensor_data["reading"],
            sensor_data["temperature"],
            sensor_data["vibration_rms"],
            sensor_data["door_event"],
            sensor_data["anomaly"]
        ])

client = mqtt.Client()
try:
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    print(f"Connected to MQTT Broker on {BROKER}:{PORT}")
except Exception as e:
    print(f"MQTT connection warning: {e}. Outputting locally to CSV only.")

initialize_csv()

print("=" * 70)
print(f"Starting Simulation | Mode: {args.anomaly} | Duration: {args.minutes} min")
print("=" * 70)

for reading in range(1, TOTAL_SECONDS + 1):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    door_event = "OPEN" if random.random() < 0.10 else "CLOSE"

    # Temperature Stream (1 Hz)
    # NOTE: drift rate reduced from 0.08 to 0.008 degC/reading, plus a hard
    # safety cap, after the original unbounded 0.08 rate was found to produce
    # physically impossible temperatures (76 degC at 15 min, 148 degC at 30
    # min) that then exploded into extreme z-scores once normalized against
    # a calm ~4 degC Normal-class baseline -- making anomaly classification
    # trivially easy rather than a genuine learned pattern, and invalidating
    # benchmark accuracy as a meaningful number. A refrigeration failure
    # drifts toward ambient temperature, not toward infinity; MAX_DRIFT_TEMP_C
    # reflects that physically, independent of how long a simulation runs.
    MAX_DRIFT_TEMP_C = 30.0
    if args.anomaly in ["temp_drift", "combined"]:
        ramped_temp = min(4.0 + (reading * 0.008), MAX_DRIFT_TEMP_C)
        temperature = round(ramped_temp + random.normalvariate(0, 0.1), 2)
    else:
        temperature = round(random.normalvariate(4.0, 0.3), 2)

    # Vibration Stream (0.5 Hz -> output valid sample every second for pipeline)
    if args.anomaly in ["vibration", "combined"]:
        vibration = round(random.normalvariate(1.2, 0.15), 2)
    else:
        vibration = round(random.normalvariate(0.45, 0.05), 2)

    sensor_data = {
        "truck_id": "truck_001",
        "reading": reading,
        "timestamp": timestamp,
        "temperature": temperature,
        "vibration_rms": vibration,
        "door_event": door_event,
        "anomaly": args.anomaly
    }

    payload = json.dumps(sensor_data)
    try:
        client.publish(TOPIC, payload)
    except Exception:
        pass

    save_to_csv(sensor_data)
    if reading % 10 == 0 or reading == 1:
        print(f"[{timestamp}] Reading #{reading:<4} Temp: {temperature:5.2f}°C | Vib: {vibration:4.2f}g | Door: {door_event}")
    
    if not args.fast:
        time.sleep(1)

try:
    client.loop_stop()
    client.disconnect()
except Exception:
    pass

print("\nSimulation completed successfully.")