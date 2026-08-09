import json
import csv
import os
import paho.mqtt.client as mqtt

# ==========================================================
# MQTT Configuration
# ==========================================================
BROKER = "localhost"
PORT = 1883
TOPIC = "logibridge/truck_001/sensors"

# ==========================================================
# Output CSV
# ==========================================================
CSV_FILE = "processed_data.csv"

# ==========================================================
# Required JSON fields
# ==========================================================
REQUIRED_FIELDS = [
    "truck_id",
    "reading",
    "timestamp",
    "temperature",
    "vibration_rms",
    "door_event",
    "anomaly"
]


# ==========================================================
# Save Data to CSV
# ==========================================================
def save_to_csv(data):

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "truck_id",
                "reading",
                "temperature",
                "vibration",
                "door",
                "anomaly"
            ])

        writer.writerow([
            data["timestamp"],
            data["truck_id"],
            data["reading"],
            data["temperature"],
            data["vibration_rms"],
            data["door_event"],
            data["anomaly"]
        ])

    print("Saved to processed_data.csv")


# ==========================================================
# Validate JSON
# ==========================================================
def validate_data(data):

    for field in REQUIRED_FIELDS:
        if field not in data:
            print(f"Missing field: {field}")
            return False

    return True


# ==========================================================
# When Connected
# ==========================================================
def on_connect(client, userdata, flags, rc):

    if rc == 0:
        print("\nConnected to MQTT Broker.")
        client.subscribe(TOPIC)
        print(f"Subscribed to topic: {TOPIC}")
    else:
        print("Connection failed")


# ==========================================================
# When Message Arrives
# ==========================================================
def on_message(client, userdata, msg):

    print("\n" + "=" * 70)

    payload = msg.payload.decode()

    print("Raw MQTT Message:")
    print(payload)

    try:

        data = json.loads(payload)

        print("\nParsed JSON:")
        print(data)

        if validate_data(data):

            print("\nVALID DATA")

            save_to_csv(data)

        else:

            print("\nINVALID DATA")

    except json.JSONDecodeError:

        print("\nInvalid JSON received.")


# ==========================================================
# Main
# ==========================================================
client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

print("Waiting for sensor data...\n")

client.connect(BROKER, PORT, 60)

client.loop_forever()
