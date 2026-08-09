"""
=========================================================
LogiBridge - Real-Time Output-Confidence Drift Monitor
Module 5 - Task E1
=========================================================
Two modes:

  --mode build-reference
      Subscribes to the inference topic, collects confidence scores from
      a CLEAN Normal-only run (--reference-samples, default 300), bins
      them into 4 confidence buckets ([0,.25) [.25,.5) [.5,.75) [.75,1]),
      and saves reference_dist.json.

  --mode monitor (default)
      Subscribes continuously, keeps a rolling window of the last
      --window-size (default 100) confidence scores, recomputes PSI
      against reference_dist.json every --check-interval seconds
      (default 60), prints the current PSI, and -- when PSI exceeds
      --threshold (default 0.25) -- prints the exact alert string
      required by the brief and publishes an alert to the alerts topic.
=========================================================
"""

import argparse
import json
import os
import threading
import time
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt

BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]


def bucketize(confidences, bin_edges=BIN_EDGES):
    counts, _ = np.histogram(confidences, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return (counts / total).tolist()


def calculate_psi(ref_pct, actual_pct):
    ref_arr = np.where(np.array(ref_pct) == 0, 1e-4, ref_pct)
    actual_arr = np.where(np.array(actual_pct) == 0, 1e-4, actual_pct)
    return float(np.sum((actual_arr - ref_arr) * np.log(actual_arr / ref_arr)))


class DriftMonitor:
    def __init__(self, args):
        self.args = args
        self.window = deque(maxlen=args.window_size)
        self.build_buffer = []
        self.reference_pct = None
        self.lock = threading.Lock()

        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.inference_topic = f"logibridge/trucks/{args.truck_id}/inference"
        self.alerts_topic = f"logibridge/trucks/{args.truck_id}/alerts"

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"\nConnected to MQTT Broker. Subscribing to {self.inference_topic}")
            client.subscribe(self.inference_topic)
        else:
            print(f"Connection failed, rc={rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            confidence = float(payload["confidence"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return

        with self.lock:
            if self.args.mode == "build-reference":
                self.build_buffer.append(confidence)
                if len(self.build_buffer) % 25 == 0:
                    print(f"[BUILD-REFERENCE] Collected {len(self.build_buffer)}/{self.args.reference_samples} clean samples...")
                if len(self.build_buffer) >= self.args.reference_samples:
                    self._save_reference()
                    client.disconnect()
            else:
                self.window.append(confidence)
                self._total_received = getattr(self, "_total_received", 0) + 1

    def _save_reference(self):
        pct = bucketize(self.build_buffer[: self.args.reference_samples])
        ref = {
            "bin_edges": BIN_EDGES,
            "bin_percentages": pct,
            "n_samples": self.args.reference_samples,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.args.ref)) or ".", exist_ok=True)
        with open(self.args.ref, "w", encoding="utf-8") as f:
            json.dump(ref, f, indent=4)
        print(f"\n[BUILD-REFERENCE] Saved reference distribution to: {self.args.ref}")
        print(f"[BUILD-REFERENCE] Bin percentages {BIN_EDGES}: {pct}")

    def _monitor_loop(self):
        with open(self.args.ref, "r", encoding="utf-8") as f:
            ref = json.load(f)
        self.reference_pct = ref["bin_percentages"]

        print(f"[MONITOR] Watching {self.inference_topic} | rolling window={self.args.window_size} "
              f"| check every {self.args.check_interval}s | alert threshold PSI>{self.args.threshold}")

        last_seen_count = 0  # track total messages received to detect stale window

        while True:
            time.sleep(self.args.check_interval)
            with self.lock:
                sample_count = len(self.window)
                current_total = getattr(self, "_total_received", 0)

                if sample_count == 0:
                    print("[MONITOR] No inferences received yet.")
                    continue

                # Skip PSI check if no NEW messages arrived since last check
                if current_total == last_seen_count:
                    print(f"[MONITOR] No new inferences in last {self.args.check_interval}s — skipping PSI check.")
                    continue

                last_seen_count = current_total
                actual_pct = bucketize(list(self.window))

            psi_value = calculate_psi(self.reference_pct, actual_pct)
            print(f"[MONITOR] PSI={psi_value:.3f} (window={sample_count} samples)")

            if psi_value > self.args.threshold:
                alert_msg = f"[LOGIBRIDGE DRIFT ALERT] PSI={psi_value:.3f}"
                print(alert_msg)
                alert_payload = json.dumps({
                    "truck_id": self.args.truck_id,
                    "alert_type": "DRIFT",
                    "psi_value": round(psi_value, 3),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                self.client.publish(self.alerts_topic, alert_payload, qos=1)

    def run(self):
        print(f"Connecting to MQTT broker at {self.args.broker}:{self.args.port}...")
        self.client.connect(self.args.broker, self.args.port, 60)

        if self.args.mode == "build-reference":
            self.client.loop_forever()
        else:
            if not os.path.exists(self.args.ref):
                raise FileNotFoundError(
                    f"Reference distribution not found at {self.args.ref}. "
                    f"Run with --mode build-reference on a clean Normal-only run first."
                )
            self.client.loop_start()
            try:
                self._monitor_loop()
            except KeyboardInterrupt:
                print("\nStopping Drift Monitor.")
                self.client.loop_stop()
                self.client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiBridge Output-Confidence Drift Monitor")
    parser.add_argument("--mode", choices=["monitor", "build-reference"], default="monitor")
    parser.add_argument("--broker", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--truck-id", type=str, default="truck_001")
    parser.add_argument("--ref", type=str, default="reference_dist.json", help="Path to reference_dist.json")
    parser.add_argument("--window-size", type=int, default=100, help="Rolling window of inferences")
    parser.add_argument("--check-interval", type=int, default=10, help="Seconds between PSI checks")
    parser.add_argument("--threshold", type=float, default=0.25, help="PSI drift alert threshold")
    parser.add_argument("--reference-samples", type=int, default=300, help="Samples to collect in build-reference mode")
    args = parser.parse_args()

    DriftMonitor(args).run()
