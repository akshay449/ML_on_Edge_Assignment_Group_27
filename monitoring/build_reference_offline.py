"""
=========================================================
LogiBridge - Offline Reference Distribution Builder
Module 5 - Task E1 (helper)
=========================================================
Builds monitoring/reference_dist.json directly from the Normal-class
rows already present in training/training_dataset.csv (produced by
generate_dataset.py), by running them through the trained TFLite model
and binning the resulting confidence scores into 4 buckets -- the same
format drift_monitor.py expects.

This is a fast offline alternative to:
    python drift_monitor.py --mode build-reference
which requires ~50 minutes of live MQTT traffic to collect 300 samples.
Both produce an identical reference_dist.json -- use whichever suits
your demo. (For the E1 demo video you'll still want to show the live
`--mode monitor` + drift injection, just not necessarily the 50-minute
reference-collection step.)
=========================================================
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "data_pipeline")
if DATA_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, DATA_PIPELINE_DIR)

from preprocessing import FEATURE_COLUMNS

NORM_FEATURE_COLUMNS = [f"{c}_norm" for c in FEATURE_COLUMNS]
BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]


def find_model_path():
    """Same priority order as inference_service.py: pruned+quantised (M3)
    > PTQ-only (M2) > FP32 baseline (M1)."""
    candidates = [
        os.path.join(PROJECT_ROOT, "models", "logibridge_pruned_int8.tflite"),
        os.path.join(PROJECT_ROOT, "models", "logibridge_int8.tflite"),
        os.path.join(PROJECT_ROOT, "models", "cold_chain_model.tflite"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    tflite_files = glob.glob(os.path.join(PROJECT_ROOT, "models", "*.tflite"))
    if tflite_files:
        return tflite_files[0]
    raise FileNotFoundError(
        "No .tflite model found under models/. Run train_model.py / convert_ptq.py "
        "/ prune_quantise.py first."
    )


def quantize_input(x_float, input_details):
    dtype = input_details["dtype"]
    if dtype in (np.int8, np.uint8):
        scale, zero_point = input_details.get("quantization", (1.0, 0))
        if not scale:
            scale = 1.0
        q = x_float / scale + zero_point
        return np.clip(np.round(q), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return x_float.astype(np.float32)


def dequantize_output(y_raw, output_details):
    dtype = output_details["dtype"]
    if dtype in (np.int8, np.uint8):
        scale, zero_point = output_details.get("quantization", (1.0, 0))
        if not scale:
            scale = 1.0
        return (y_raw.astype(np.float32) - zero_point) * scale
    return y_raw.astype(np.float32)


def bucketize(confidences, bin_edges=BIN_EDGES):
    counts, _ = np.histogram(confidences, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return (counts / total).tolist()


def build_reference(dataset_path, model_path, output_path):
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    if "label" not in df.columns:
        raise ValueError(f"{dataset_path} has no 'label' column.")

    normal_df = df[df["label"] == 0]
    if len(normal_df) == 0:
        raise ValueError(f"No Normal-class (label==0) rows found in {dataset_path}.")

    missing = [c for c in NORM_FEATURE_COLUMNS if c not in normal_df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}. "
            f"Re-run generate_dataset.py with the current preprocessing.py."
        )

    print(f"Found {len(normal_df)} Normal-class windows.")

    print(f"Loading TFLite model from: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    confidences = []
    X = normal_df[NORM_FEATURE_COLUMNS].values.astype(np.float32)
    for i in range(len(X)):
        sample = np.expand_dims(X[i], axis=0)
        q_sample = quantize_input(sample, input_details)
        interpreter.set_tensor(input_details["index"], q_sample)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_details["index"])
        out = dequantize_output(raw_out, output_details)
        confidences.append(float(np.max(out)))

    pct = bucketize(confidences)
    ref = {
        "bin_edges": BIN_EDGES,
        "bin_percentages": pct,
        "n_samples": len(confidences),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=4)

    print(f"\nSaved reference distribution ({len(confidences)} samples) to: {output_path}")
    print(f"Bin percentages {BIN_EDGES}: {pct}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build reference_dist.json offline from existing training data (no live MQTT wait needed)"
    )
    parser.add_argument(
        "--dataset", type=str,
        default=os.path.join(PROJECT_ROOT, "training", "training_dataset.csv"),
        help="Path to training_dataset.csv (must contain a 'label' column with 0=Normal rows)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to .tflite model (default: auto-detect best available, same priority as inference_service.py)",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(SCRIPT_DIR, "reference_dist.json"),
        help="Where to write reference_dist.json",
    )
    args = parser.parse_args()

    resolved_model_path = args.model or find_model_path()
    build_reference(args.dataset, resolved_model_path, args.output)
