"""
=========================================================
LogiBridge - Post-Training Quantization (PTQ) Pipeline (M2)
Module 4 - Task D1
=========================================================
Converts Keras baseline model to full INT8 TFLite model (M2)
using representative dataset quantization.
=========================================================
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

# Absolute path to current script location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PIPELINE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data_pipeline"))
if DATA_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, DATA_PIPELINE_DIR)

from preprocessing import FEATURE_COLUMNS

NORM_FEATURE_COLUMNS = [f"{c}_norm" for c in FEATURE_COLUMNS]


def locate_file(file_path):
    """Finds input files across potential relative paths."""
    possible_paths = [
        file_path,
        os.path.join(CURRENT_DIR, "..", file_path),
        os.path.join(CURRENT_DIR, file_path),
        os.path.join("models", "saved_model.keras"),
        os.path.join("..", "models", "saved_model.keras"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not locate target file: {file_path}")


def representative_dataset_gen():
    """Generates calibration samples for INT8 quantization scale/zero-point parameters.

    Uses training_dataset.csv (the full 3-class combined dataset from
    generate_dataset.py), TRAIN split only -- NOT clean_data.csv, which only
    ever contains whichever single class generate_dataset.py processed last
    (Critical, in the current pipeline order). Calibrating INT8 scale/zero-
    point on a single class's value range distorts quantization for inputs
    outside that range -- this was found to be the cause of M2 (plain PTQ,
    otherwise identical to the working M1 float32 weights) collapsing to
    ~70% accuracy while M1 itself scored ~94%. This mirrors the identical
    fix already applied to prune_quantise.py's representative_dataset_gen.
    """
    data_path = locate_file(os.path.join("training", "training_dataset.csv"))
    df = pd.read_csv(data_path)
    if "split" in df.columns:
        df = df[df["split"] == "train"]

    missing = [c for c in NORM_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"training_dataset.csv is missing expected columns {missing}. "
            f"Re-run generate_dataset.py with the current preprocessing.py."
        )

    X = df[NORM_FEATURE_COLUMNS].values.astype(np.float32)

    # Yield calibration samples one by one (>=100, per Task F1 needs >=200 overall
    # across the pipeline's representative-dataset calls)
    for i in range(min(200, len(X))):
        yield [np.expand_dims(X[i], axis=0)]


def run_ptq(keras_model_path="models/saved_model.keras", output_dir="models"):
    resolved_model_path = locate_file(keras_model_path)
    print(f"Loading Keras baseline model from: {resolved_model_path}")
    model = tf.keras.models.load_model(resolved_model_path)

    # Initialize TFLite Converter from Keras Model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enforce full INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting model to INT8 Post-Training Quantization (M2)...")
    tflite_quant_model = converter.convert()

    os.makedirs(output_dir, exist_ok=True)

    # Save output artifact (this is the filename inference_service.py and
    # benchmark.py both look for as the M2 variant)
    output_path = os.path.join(output_dir, "logibridge_int8.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Successfully saved INT8 PTQ model (M2) to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiBridge Post-Training Quantization Pipeline")
    parser.add_argument("--model", type=str, default="models/saved_model.keras", help="Path to input Keras model")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save output models")
    args = parser.parse_args()

    run_ptq(args.model, args.output_dir)
