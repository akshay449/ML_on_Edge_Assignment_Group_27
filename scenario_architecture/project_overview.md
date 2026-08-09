# LogiEdge — System Architecture Overview

```
+-----------------------------------------------------------------------------------+
|                          LOGIEDGE SYSTEM ARCHITECTURE                             |
+-----------------------------------------------------------------------------------+

 [Sensors: Temp (1Hz), Vibration (0.5Hz), Door (event)]
             |
             v  Raw JSON over local MQTT: logibridge/truck_001/sensors  (QoS 0)
 +-----------------------+
 | Mosquitto MQTT Broker |   <- runs locally on the truck edge node; raw telemetry
 +-----------------------+      never leaves the vehicle
             |
             v
 +-----------------------+
 | MQTT Consumer Engine  | ---> validates JSON, stores raw stream in sensor_data.csv
 +-----------------------+
             |
             v
 +----------------------------------------+
 | Preprocessing Pipeline                  |
 |  [PLANNED] 5-sample moving average       |
 |  [PLANNED] 30s window / 10s step         |
 |  [PLANNED] 6 features: temp_mean,        |
 |    temp_std, temp_slope, vib_rms,        |
 |    vib_peak, vib_kurtosis                |
 |  [CURRENT] row-level z-score normalise   |  <- see Known Limitations
 +----------------------------------------+
             |
             v   clean_data.csv (feature vectors) + training_stats.json (frozen
             |   from Normal-class only, loaded — never recomputed live)
             v
 +-----------------------+
 |  TFLite INT8 Engine   | ---> Local classification: Normal(0) / Warning(1) / Critical(2)
 +-----------------------+
             |
             +-----------------------------------+
             |                                   |
             v  logibridge/trucks/truck_001/      v  rolling confidence scores
             |  inference  (QoS 1)                |
 +-----------------------+           +---------------------------------+
 |  MQTT Alert Publisher |           | PSI Drift Monitor                |
 |  (predicted_class,    |           | on OUTPUT confidence distribution|
 |   confidence)         |           | (4 bins); alert if PSI > 0.25    |
 +-----------------------+           +---------------------------------+
                                                    |
                                                    v
                                      logibridge/trucks/truck_001/alerts
                                      [PLANNED — not yet published]
                                                    |
                                                    v
                                        Operations Centre (backfilled on
                                        reconnect after cellular outages)
```

---

## Cellular boundary

Everything above the `Mosquitto MQTT Broker` line runs entirely on the truck. Only two
things are meant to cross the cellular uplink to the operations centre:

1. Inference results (`logibridge/trucks/truck_001/inference`)
2. Alerts (`logibridge/trucks/truck_001/alerts`)

Raw 1 Hz telemetry stays local — this is what delivers the ~99.7% bandwidth reduction
quantified in `constraint_analysis.md`.

---

## Known Limitations (see README for details)

- Preprocessing does not yet implement the moving-average / windowed feature
  extraction the architecture above assumes — it currently normalises raw per-second
  readings instead.
- `training_stats.json` is currently recomputed on every preprocessing run rather
  than frozen once from 10 minutes of Normal-class data.
- The Drift Monitor currently evaluates PSI on raw input features, not the model's
  output confidence distribution, and does not publish to the alerts topic.
- Docker containerisation and the Ansible deployment playbook are not yet present in
  this repository.
