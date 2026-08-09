# LogiEdge – AI-Powered Cold Chain Monitoring System

## Project Overview

LogiEdge is an Edge AI system for cold-chain fleet monitoring. It simulates
multi-sensor telemetry (temperature, vibration, door status), transports it over
local MQTT, preprocesses it into feature vectors, classifies cargo state on-device
with a compressed TFLite model (Normal / Warning / Critical), monitors for data
drift, and benchmarks model variants for deployment.

Built for the AIML ZG535 – Machine Learning on Edge mini-project (FreightBridge
Logistics 85-truck cold-chain pilot). See `constraint_analysis.md` for the
quantified latency/bandwidth/connectivity/privacy case, and
`hardware_justification.md` for the hardware selection and Roofline analysis.

---

# Project Structure

```
logibridge/
├── README.md
├── requirements.txt
├── scenario_architecture/
│   ├── constraint_analysis.md
│   └── project_overview.md
├── hardware/
│   └── hardware_justification.md
├── data_pipeline/
│   ├── simulator.py
│   ├── consumer.py
│   ├── preprocessing.py
│   ├── training_stats.json
│   └── mqtt_architecture.md
├── training/
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── convert_ptq.py
│   ├── prune_quantise.py
│   └── models/
├── inference/
│   ├── Dockerfile              # not yet present — see Known Limitations
│   └── inference_service.py
├── monitoring/
│   └── drift_monitor.py
├── deployment/
│   └── logibridge_deploy.yml   # not yet present — see Known Limitations
├── optimisation/
│   ├── benchmark.py
│   └── results/
└── reports/
```

---

# Requirements

```
tensorflow==2.15.*
tensorflow-model-optimization
pandas
numpy
scikit-learn
matplotlib
paho-mqtt<2.0        # v2 changes the callback signature used in this codebase
psutil                # needed for energy-per-inference benchmarking (see Known Limitations)
```

Save as `requirements.txt` and `pip install -r requirements.txt`.

---

# How to Run

## 0. Prerequisites

- Python 3.11+
- Mosquitto broker running locally: `mosquitto -v` (or `sudo systemctl start mosquitto`)
- `pip install -r requirements.txt`

## 1. One-time: build the labelled training dataset

`generate_dataset.py` drives the simulator directly (writing straight to
`sensor_data.csv`, no live MQTT consumer needed for this step) for each class in
turn, then runs preprocessing and stitches the labelled dataset together:

```bash
cd training
python generate_dataset.py
```

This produces `training/training_dataset.csv` (Normal / Warning / Critical windows)
and, as a side effect, `data_pipeline/training_stats.json` and
`data_pipeline/clean_data.csv`.

> ⚠️ **Known issue:** because `preprocess_data()` is called once per class inside
> this script, `training_stats.json` currently gets overwritten by whichever class
> ran last, instead of being frozen from Normal-class data only. Fix this before
> relying on the 3σ-shift experiment (step 6) or the drift monitor (step 7).

## 2. Train the baseline model (M1)

```bash
python train_model.py --data ../training/training_dataset.csv --model-dir ../models
```

Produces `models/saved_model.keras` and `models/cold_chain_model.tflite` (FP32
baseline). Checks Class 2 recall against the 95% gate and prints a warning if below.

## 3. Quantise (M2) and prune+quantise (M3)

```bash
python convert_ptq.py --model ../models/saved_model.keras --output-dir ../models
python prune_quantise.py --data ../data_pipeline/clean_data.csv --output-dir ../models
```

- `convert_ptq.py` → `models/logibridge_int8.tflite` (M2)
- `prune_quantise.py` → `models/cold_chain_model_pruned.tflite` (M3)

> ⚠️ **Known issue:** `inference_service.py` and `benchmark.py` look for
> `models/logibridge_pruned_int8.tflite`, which `prune_quantise.py` never creates.
> Either rename the output in `prune_quantise.py` or update the search paths in both
> consumers so M3 is actually picked up.

## 4. Benchmark all three variants

```bash
cd ../optimisation
python benchmark.py
```

Currently reports size and latency only; accuracy and energy-per-inference (Task F2)
still need to be added — see Known Limitations.

## 5. Live pipeline demo (simulator → broker → consumer → inference)

Run each in its own terminal, broker already running:

```bash
# Terminal 1
cd data_pipeline
python consumer.py

# Terminal 2
python simulator.py --anomaly none --minutes 5
# then repeat with --anomaly temp_drift / vibration / combined to see anomaly classes

# Terminal 3
cd ../inference
python inference_service.py
```

`inference_service.py` auto-locates the best available model
(`logibridge_pruned_int8.tflite` → `logibridge_int8.tflite` →
`cold_chain_model.tflite`, in that priority order) and prints live predictions.

## 6. Normalisation 3σ-shift experiment (Task C2 mandatory experiment)

```bash
cd ../data_pipeline
python preprocessing.py --input sensor_data.csv --sigma-shift 3.0
```

Compare model accuracy on the shifted output vs. the unshifted baseline and report
both in the Phase 2 report.

## 7. Drift monitoring demo

```bash
cd ../monitoring
python drift_monitor.py --ref ../data_pipeline/training_stats.json --curr ../data_pipeline/clean_data.csv
```

⚠️ Currently monitors raw feature drift, not the required output confidence-score
PSI — see `mqtt_architecture.md` §5 before using this for the E1 demo video.

## 8. Docker / Ansible (Task D2 / E2)

Not yet implemented in this repo. Both are mandatory deliverables:
- `inference/Dockerfile` — base `python:3.11-slim`, all `pip install` layers before
  `COPY model.tflite .` so an OTA model swap only invalidates the last layer;
  container must read `MODEL_PATH` env var and publish to
  `logibridge/trucks/{truck_id}/inference`.
- `deployment/logibridge_deploy.yml` — exactly 7 Ansible tasks (create dir, copy
  model, copy `reference_dist.json`, stop container, pull image, start container,
  wait+verify) — see the problem statement Task E2 for the exact task list.

---

# Known Limitations (fix before final submission)

1. **Preprocessing** does not implement the 5-sample MA filter + 30s/10s windowed
   6-feature extraction the assignment requires (`temp_mean`, `temp_std`,
   `temp_slope`, `vib_rms`, `vib_peak`, `vib_kurtosis`) — it normalises raw
   per-second readings instead.
2. **`training_stats.json`** is recomputed per class run instead of frozen once from
   10 minutes of Normal-class data.
3. **`drift_monitor.py`** watches raw input feature drift, not the output
   confidence-score distribution (4 bins) the brief specifies, and doesn't publish
   to an alerts topic or use the exact `[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}`
   format yet.
4. **Model filename mismatch:** `prune_quantise.py` output name doesn't match what
   `inference_service.py`/`benchmark.py` search for.
5. **`benchmark.py`** measures 2 of the 5 required metrics (size, latency) — missing
   accuracy and energy-per-inference.
6. **Docker and Ansible** deliverables are not present yet.
7. **`paho-mqtt` v2 compatibility:** pin `<2.0` or update callback signatures.
8. **Roofline / Arithmetic Intensity (Task B2)** analysis is now written up in
   `hardware_justification.md` §3 but has no accompanying code artifact — purely a
   documentation task per the brief, no action needed beyond the report.

---

# Author

Linda Kuriakose — AIML ZG535, Machine Learning on Edge, BITS Pilani WILP
