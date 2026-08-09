# MQTT Architecture

## Overview

LogiEdge uses MQTT (Mosquitto broker, running locally on each truck's edge node) for
lightweight, decoupled communication between the sensor simulator, the consumer, and
the inference service. The broker is local — only inference/alert output is intended
to leave the truck over the cellular uplink (see `constraint_analysis.md` §3).

---

# 1. Topic Hierarchy

| Purpose | Topic | Implemented in |
|---|---|---|
| Raw sensor telemetry | `logibridge/truck_001/sensors` | `simulator.py` (publish), `consumer.py` (subscribe) |
| Edge inference output | `logibridge/trucks/truck_001/inference` | `inference_service.py` (publish) |
| Drift / status alerts | `logibridge/trucks/truck_001/alerts` | **Not yet implemented** — see §5 |

> **Naming note:** the raw telemetry topic uses `truck_001/sensors` (no `trucks/`
> segment) while inference/alerts use `trucks/truck_001/...`. This is intentional in
> the current code (`simulator.py`/`consumer.py` vs `inference_service.py`) but is a
> naming inconsistency worth normalising to `logibridge/trucks/truck_001/sensors`
> before final submission, so the whole hierarchy nests under `trucks/{id}/...`.

---

# 2. QoS

- **Telemetry (`.../sensors`):** QoS 0 — high frequency (1 Hz), loss-tolerant since
  each reading feeds a smoothing/windowing stage; an occasional dropped sample does
  not change the windowed feature output materially.
- **Inference/alerts (`.../inference`, `.../alerts`):** QoS 1 — low frequency,
  operationally significant; at-least-once delivery is required so a Critical alert
  is never silently lost. (Current code publishes with the client default, QoS 0 —
  should be set explicitly to `qos=1` on these two topics before final submission.)

---

# 3. Payload Schemas

**A. Raw Sensor Payload** (`logibridge/truck_001/sensors`)
```json
{
  "truck_id": "truck_001",
  "reading": 120,
  "timestamp": "2026-08-05 22:30:00",
  "temperature": 4.12,
  "vibration_rms": 0.46,
  "door_event": "CLOSE",
  "anomaly": "none"
}
```

**B. Inference Output Payload** (`logibridge/trucks/truck_001/inference`)
```json
{
  "truck_id": "truck_001",
  "window_index": 12,
  "predicted_class": 0,
  "confidence": 0.9842
}
```

**C. Alert Payload** (`logibridge/trucks/truck_001/alerts`) — proposed schema, not
yet published by any script:
```json
{
  "truck_id": "truck_001",
  "alert_type": "DRIFT" ,
  "psi_value": 0.312,
  "timestamp": "2026-08-05 22:31:00"
}
```

---

# 4. Components

- **Mosquitto Broker:** local communication hub; receives published sensor data,
  delivers to subscribers.
- **Sensor Simulator (`simulator.py`):** publishes temperature, vibration, door, and
  anomaly-mode telemetry to `.../sensors`.
- **MQTT Consumer (`consumer.py`):** subscribes to `.../sensors`, validates the JSON
  schema, writes to `processed_data.csv`.
- **Preprocessing (`preprocessing.py`):** reads collected telemetry, cleans it, and
  (once corrected — see Known Gap below) produces windowed feature vectors.
- **Inference Service (`inference_service.py`):** runs TFLite inference on windowed
  features, publishes classification + confidence to `.../inference`.

---

# 5. Known Gap — Alerts Topic and Drift Monitoring

The project brief (Task E1) requires PSI drift monitoring on the **model's output
confidence-score distribution** (4 bins), computed on a rolling window of the last
100 inferences every 60 seconds, with alerts printed as
`[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}` and (implicitly, per the architecture)
published to an alerts topic.

The current `drift_monitor.py` instead computes PSI on raw input feature
distributions (temperature/vibration) against `training_stats.json`, and never
publishes to MQTT at all — it only prints to stdout. Before final submission this
should be reworked to:
1. Build `reference_dist.json` from 300 clean Normal-class inference confidence
   scores, binned into `[0,0.25), [0.25,0.50), [0.50,0.75), [0.75,1.0]`.
2. Maintain a rolling buffer of the last 100 live inference confidences.
3. Recompute PSI every 60 s against the reference distribution.
4. Print the exact alert string above when PSI > 0.25, and publish an alert payload
   (schema C above) to `logibridge/trucks/truck_001/alerts`.

---

# 6. Data Flow

```
Temperature / Vibration / Door sensors
        │
        ▼
 Sensor Simulator  ──publish──▶  logibridge/truck_001/sensors
        │                              │
        │                       Mosquitto Broker
        │                              │
        │                        MQTT Consumer ──▶ processed_data.csv
        │
        ▼
 Preprocessing (MA filter → 30s/10s windowing → 6 features)  [gap: not yet windowed]
        │
        ▼
 TFLite Inference Service
        │
        ├──publish──▶ logibridge/trucks/truck_001/inference
        │
        ▼
 PSI Drift Monitor (on confidence distribution)  [gap: currently on raw features]
        │
        └──publish──▶ logibridge/trucks/truck_001/alerts   [gap: not yet published]
```

---

# Conclusion

The pub/sub skeleton (simulator → broker → consumer → inference) is implemented and
consistent with the architecture. The two gaps that need closing before final
submission are the alerts topic (never published to) and the drift monitor's target
distribution (raw features vs. required confidence-score bins).
