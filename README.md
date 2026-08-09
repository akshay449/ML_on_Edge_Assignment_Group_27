# LogiEdge – AI-Powered Cold Chain Monitoring System

## Project Overview

LogiEdge is an Edge AI system for cold-chain fleet monitoring. It simulates
multi-sensor telemetry (temperature, vibration, door status), transports it over
local MQTT, preprocesses it into feature vectors, classifies cargo state on-device
with a compressed TFLite model (Normal / Warning / Critical), monitors for output
confidence-score drift, and benchmarks model variants for deployment.

Built for the AIML ZG535 – Machine Learning on Edge mini-project (FreightBridge
Logistics 85-truck cold-chain pilot). See `scenario_architecture/constraint_analysis.md`
for the quantified latency/bandwidth/connectivity/privacy case, and
`hardware/hardware_justification.md` for the hardware selection and Roofline analysis.

---

## Project Structure

```
ML_on_Edge_Assignment_Group_27/
├── README.md
├── data_pipeline/
│   ├── simulator.py            # MQTT sensor simulator (temp, vibration, door events)
│   ├── consumer.py             # MQTT subscriber — writes raw telemetry to CSV
│   ├── preprocessing.py        # 5-sample MA → 30s/10s windowed 6-feature extraction + Z-score normalisation
│   ├── training_stats.json     # Frozen Normal-class baseline stats for Z-score normalisation
│   ├── sensor_data.csv         # Raw simulator output
│   ├── clean_data.csv          # Preprocessed Normal-class data
│   ├── processed_data.csv      # Consumer-written output
│   └── mqtt_architecture.md    # MQTT topic/namespace design
├── training/
│   ├── generate_dataset.py     # Runs simulator for each class, builds training_dataset.csv
│   ├── train_model.py          # Trains M1 (FP32 baseline), checks recall/accuracy gates
│   ├── convert_ptq.py          # Post-training INT8 quantisation → M2
│   ├── prune_quantise.py       # Magnitude pruning + INT8 quantisation → M3
│   ├── training_dataset.csv    # Labelled windowed feature dataset (Normal/Warning/Critical)
│   └── models/                 # Training-time model artefacts
├── models/                     # Deployment-ready model artefacts
│   ├── saved_model.keras        # M1 — Keras FP32 baseline
│   ├── cold_chain_model.tflite  # M1 — FP32 TFLite
│   ├── logibridge_int8.tflite   # M2 — PTQ INT8
│   ├── logibridge_pruned_int8.tflite  # M3 — Pruned + PTQ INT8
│   ├── baseline_weights.npz
│   ├── validation_split.npz    # Held-out validation split for consistent benchmarking
│   └── training_summary.json
├── inference/
│   ├── inference_service.py    # Real-time TFLite inference over MQTT (auto-selects best model)
│   ├── Dockerfile              # Multi-layer Docker image (OTA-swap optimised layer order)
│   └── requirements.txt
├── monitoring/
│   ├── drift_monitor.py        # Real-time PSI drift monitor on output confidence scores
│   ├── build_reference_offline.py  # Offline reference distribution builder (faster alternative)
│   └── reference_dist.json     # Reference confidence-score distribution (4 bins)
├── optimisation/
│   ├── benchmark.py            # Five-metric benchmark: latency, p95, size, accuracy, energy/inference
│   └── optimisation/results/   # benchmark_results.csv and benchmark_results_detailed.json
├── deployment/
│   ├── logibridge_deploy.yml   # Ansible playbook — 7-task OTA deployment to edge nodes
│   └── inventory.ini
├── hardware/
│   └── hardware_justification.md
├── scenario_architecture/
│   ├── constraint_analysis.md
│   ├── project_overview.md
│   └── *.png                   # Architecture and workflow diagrams
└── reports/
    ├── Phase1_Report.docx
    ├── Phase2_Report.docx
    └── Final_Report.docx
```

---

## Requirements

```
tensorflow==2.15.*
tensorflow-model-optimization
pandas
numpy
scipy
scikit-learn
matplotlib
paho-mqtt<2.0
psutil
```

Install with:

```bash
pip install -r inference/requirements.txt
```

---

## How to Run

### Prerequisites

- Python 3.11+
- Mosquitto broker running locally: `mosquitto -v` (or `sudo systemctl start mosquitto`)
- Install dependencies: `pip install -r inference/requirements.txt`

---

### Step 1 — Generate dataset, train and convert models

Run all commands from the `training/` directory:

```bash
cd training

# Generate the 3-class labelled dataset
python generate_dataset.py

# Train M1 (FP32 baseline) — boosts Critical-class weight to meet the 95% recall gate
python train_model.py --data ../training/training_dataset.csv --model-dir ../models --class-weight-critical 5

# Convert to M2 (PTQ INT8)
python convert_ptq.py --model ../models/saved_model.keras --output-dir ../models

# Prune + quantise to M3 (35% sparsity + INT8)
python prune_quantise.py --data ../training/training_dataset.csv --output-dir ../models --sparsity 0.35
```

Outputs:
- `models/saved_model.keras` — M1 Keras baseline
- `models/cold_chain_model.tflite` — M1 FP32 TFLite
- `models/logibridge_int8.tflite` — M2 PTQ INT8
- `models/logibridge_pruned_int8.tflite` — M3 Pruned + PTQ INT8

---

### Step 2 — Benchmark all three variants

```bash
cd optimisation
python benchmark.py --models-dir ../models --tdp 7.5
```

Reports mean latency, p95 latency, model size, held-out accuracy, Class-2 recall,
and energy per inference (using a 7.5 W TDP estimate for the edge hardware).
Results written to `optimisation/optimisation/results/`.

---

### Step 3 — Live pipeline demo

Start the MQTT broker, then run each component in a separate terminal:

```bash
# Terminal 1 — MQTT consumer (logs raw sensor JSON to processed_data.csv)
cd data_pipeline
python consumer.py

# Terminal 2 — Inference service (live 30s/10s windowing + TFLite inference)
cd inference
python inference_service.py

# Terminal 3 — Sensor simulator (drives the demo)
cd data_pipeline
python simulator.py --anomaly none --minutes 2
```

After ~30 seconds, live predictions appear in Terminal 2, all classified Normal.
Repeat the simulator with different anomaly modes to see Warning/Critical predictions:

```bash
python simulator.py --anomaly temp_drift --minutes 2
python simulator.py --anomaly vibration --minutes 2
python simulator.py --anomaly combined --minutes 2
```

`inference_service.py` auto-selects the best available model
(`logibridge_pruned_int8.tflite` → `logibridge_int8.tflite` → `cold_chain_model.tflite`)
and publishes predictions to `logibridge/trucks/{truck_id}/inference`.

---

### Step 4 — Normalisation 3σ-shift experiment

Run both baseline and shifted versions and compare model accuracy on each output:

```bash
cd data_pipeline
python preprocessing.py --input sensor_data.csv --sigma-shift 0.0   # baseline
python preprocessing.py --input sensor_data.csv --sigma-shift 3.0   # shifted
```

---

### Step 5 — Docker OTA layer-cache demo

Run from the project root:

```bash
# Build 1 — full build, all layers fresh
docker build -t logibridge-inference:v1.0 -f Dockerfile .

# Simulate an OTA model update by touching a file inside models/
python -c "import json,time; p='models/training_summary.json'; d=json.load(open(p)); d['ota_demo_timestamp']=time.time(); json.dump(d, open(p,'w'), indent=2)"

# Build 2 — only the final COPY models/ layer rebuilds; all layers above are cached
docker build -t logibridge-inference:v1.0 -f Dockerfile .

# OTA bandwidth estimate for 85-truck fleet
python -c "print(f'OTA push size for 85 trucks: {85*4.61:.1f} KB (~{85*4.61/1024:.2f} MB) -- one model file only')"
```

The Dockerfile places `COPY models/` last so an OTA model swap invalidates only
that final layer, keeping all dependency and code layers cached.

Run the container:

```bash
docker run -e MQTT_BROKER=host.docker.internal -e TRUCK_ID=truck_001 logibridge-inference:v1.0
```

---

### Step 6 — Ansible OTA deployment

```bash
# Install Ansible (WSL/Linux)
sudo apt update && sudo apt install -y ansible

cd deployment

# Run 1 — initial deploy, some tasks report "changed"
ansible-playbook -i inventory.ini logibridge_deploy.yml

# Run 2 — immediately after, no changes in between; PLAY RECAP must show changed=0
ansible-playbook -i inventory.ini logibridge_deploy.yml
```

Executes 7 idempotent tasks: create app directory, copy model, copy reference
distribution, stop container, pull image, start container, verify health.

---

### Step 7 — PSI drift monitoring demo

**Terminal 1 — start the monitor** (reduced window/interval for demo time-budget):

```bash
cd monitoring
python drift_monitor.py --mode monitor --ref reference_dist.json --window-size 20 --check-interval 15
```

**Terminal 2 — prime with clean baseline traffic, then inject drift, then recover:**

```bash
cd data_pipeline

# Prime: generates stable low PSI readings before injection
python simulator.py --anomaly none --minutes 2

# Inject: drift dominates the 20-sample window within ~1-2 minutes
python simulator.py --anomaly combined --minutes 2

# Recover: PSI drops back below threshold
python simulator.py --anomaly none --minutes 5
```

The monitor prints `[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}` and publishes to the
alerts topic when PSI exceeds 0.25. The alert should fire within 5 minutes of
injection starting.

**Build reference distribution offline** (faster alternative to the 50-minute live
collection):

```bash
cd monitoring
python build_reference_offline.py
```

---

## Authors

Group 27 — AIML ZG535, Machine Learning on Edge, BITS Pilani WILP

| Name                     | ID           |
|--------------------------|--------------|
| Linda Kuriakose          | 2024AC05557  |
| Akshay Ashok Deshpande   | 2024AD05483  |
| Kamalesh Kumar Gadhwal   | 2024AC05919  |
| A Jaya Suriya            | 2024AD05006  |
