"""
=========================================================
LogiBridge - Data Preprocessing & Windowed Feature Extraction Pipeline
Module 3 - Task C2
=========================================================
Raw telemetry (~1 Hz) -> 5-sample moving average -> 30s sliding window
(10s step) -> 6-value feature vector (+ door_open_ratio auxiliary) ->
Z-score normalisation using FROZEN Normal-class baseline stats.

This module is imported by generate_dataset.py, train_model.py,
convert_ptq.py, prune_quantise.py, and inference_service.py so every
stage of the pipeline agrees on the same FEATURE_COLUMNS, window size,
and feature-extraction math (no train/serve skew).
=========================================================
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sampling assumption (per project brief): ~1 sample/second nominal rate for
# both temperature and vibration streams once buffered on the edge node.
WINDOW_SECONDS = 30
STEP_SECONDS = 10
MA_WINDOW = 5

# Canonical feature order used by every downstream script (training, PTQ,
# pruning, live inference). Do NOT reorder without regenerating
# training_stats.json and every model.
FEATURE_COLUMNS = [
    "temp_mean",
    "temp_std",
    "temp_slope",
    "vib_rms",
    "vib_peak",
    "vib_kurtosis",
    "door_open_ratio",
]


def moving_average(series, window=MA_WINDOW):
    """5-sample moving average filter (Task C2, step 1)."""
    return series.rolling(window=window, min_periods=1, center=False).mean()


def extract_window_features(temp_window, vib_window, door_window):
    """Computes the windowed feature vector for one 30s window.

    temp_window / vib_window should already be moving-average-smoothed
    arrays. door_window is the raw 0/1 door_open array for the same window.
    Used identically by the offline batch pipeline (this file) and the
    live MQTT inference service, so training and serving never diverge.
    """
    temp_window = np.asarray(temp_window, dtype=np.float64)
    vib_window = np.asarray(vib_window, dtype=np.float64)
    door_window = np.asarray(door_window, dtype=np.float64)

    temp_mean = float(np.mean(temp_window)) if len(temp_window) else 0.0
    temp_std = float(np.std(temp_window)) if len(temp_window) else 0.0

    # Temperature rate-of-change in degC/min via linear fit over the window
    # (assumes ~1 sample/sec, so slope-per-sample * 60 = slope-per-minute).
    if len(temp_window) >= 2:
        x = np.arange(len(temp_window))
        slope_per_sample = float(np.polyfit(x, temp_window, 1)[0])
        temp_slope = slope_per_sample * 60.0
    else:
        temp_slope = 0.0

    vib_rms = float(np.sqrt(np.mean(np.square(vib_window)))) if len(vib_window) else 0.0
    vib_peak = float(np.max(np.abs(vib_window))) if len(vib_window) else 0.0

    if len(vib_window) >= 4 and np.std(vib_window) > 1e-9:
        vib_kurtosis = float(kurtosis(vib_window, fisher=True, bias=False))
    else:
        vib_kurtosis = 0.0

    door_open_ratio = float(np.mean(door_window)) if len(door_window) else 0.0

    return {
        "temp_mean": temp_mean,
        "temp_std": temp_std,
        "temp_slope": temp_slope,
        "vib_rms": vib_rms,
        "vib_peak": vib_peak,
        "vib_kurtosis": vib_kurtosis,
        "door_open_ratio": door_open_ratio,
    }


def _derive_label(features):
    """Heuristic class label from windowed features, mirroring the class
    definitions in the project brief (Normal / Warning / Critical).

    This is only a fallback / smoke-test label for standalone preprocessing
    runs. generate_dataset.py always overrides it with the ground-truth
    simulation-mode label when building the real training dataset.
    """
    temp_dev = abs(features["temp_mean"] - 4.0)
    vib_peak = features["vib_peak"]
    door_open_ratio = features["door_open_ratio"]

    if temp_dev > 3.0 or vib_peak > 1.0 or (door_open_ratio > 0.3 and temp_dev > 1.0):
        return 2
    if temp_dev > 1.0 or vib_peak > 0.7:
        return 1
    return 0


def _resolve_output_dir(output_dir):
    if not os.path.isabs(output_dir):
        if output_dir in ("data_pipeline", "."):
            return CURRENT_DIR
        return os.path.abspath(output_dir)
    return output_dir


def _load_stats(stats_path):
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_stats(stats, stats_path):
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)


def _fit_stats_from(feat_df):
    stats = {}
    for col in FEATURE_COLUMNS:
        mean_val = float(feat_df[col].mean())
        std_val = float(feat_df[col].std())
        if std_val == 0 or np.isnan(std_val):
            std_val = 1e-6
        stats[col] = {"mean": mean_val, "std": std_val}
    return stats


def preprocess_data(input_csv, output_dir, sigma_shift=0.0, fit_stats=False):
    """Runs the full MA-filter -> windowing -> feature-extraction ->
    normalisation pipeline.

    fit_stats=True computes fresh mean/std baseline stats from THIS call's
    data and OVERWRITES training_stats.json. Per Task C2 this must only be
    used on a clean Normal-class run. Every other call should leave
    fit_stats=False so the frozen baseline is reused and never recomputed
    from live/anomalous data.
    """
    output_dir = _resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "training_stats.json")

    print(f"Loading sensor telemetry from: {input_csv}")
    df = pd.read_csv(input_csv)

    temp_col = "temp_celsius" if "temp_celsius" in df.columns else "temperature"
    vib_col = "vibration_rms" if "vibration_rms" in df.columns else "vibration"

    if "door_event" in df.columns:
        df["door_open"] = (df["door_event"].astype(str).str.upper() == "OPEN").astype(int)
    elif "door_open" not in df.columns:
        df["door_open"] = 0

    # Optional 3-Sigma Mean Shift Experiment (Task C2 mandatory experiment)
    if sigma_shift != 0.0 and temp_col in df.columns:
        print(f"[EXPERIMENT] Applying {sigma_shift}-Sigma mean shift to {temp_col}.")
        temp_std_raw = df[temp_col].std()
        df[temp_col] = df[temp_col] + (sigma_shift * temp_std_raw)

    # 1) 5-sample moving average filter
    df["temp_ma"] = moving_average(df[temp_col])
    df["vib_ma"] = moving_average(df[vib_col])

    # 2) 30s sliding window / 10s step feature extraction
    n = len(df)
    rows = []
    idx = 0
    while idx + WINDOW_SECONDS <= n:
        window = df.iloc[idx: idx + WINDOW_SECONDS]
        feats = extract_window_features(
            window["temp_ma"].values, window["vib_ma"].values, window["door_open"].values
        )
        rows.append(feats)
        idx += STEP_SECONDS

    if not rows:
        # Not enough samples for a full window — degrade gracefully with a
        # single window over whatever we have rather than erroring out.
        print("[WARNING] Fewer than one full window of data available; "
              "using all available samples as a single window.")
        feats = extract_window_features(df["temp_ma"].values, df["vib_ma"].values, df["door_open"].values)
        rows.append(feats)

    feat_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    feat_df["label"] = feat_df.apply(_derive_label, axis=1)

    # 3) Normalisation using FROZEN baseline stats
    if fit_stats:
        stats = _fit_stats_from(feat_df)
        _save_stats(stats, stats_path)
        print(f"[BASELINE] Fitted and saved Normal-class reference statistics to: {stats_path}")
    elif os.path.exists(stats_path):
        stats = _load_stats(stats_path)
    else:
        print("[WARNING] training_stats.json not found — computing stats from this "
              "batch as a one-time fallback. Re-run generate_dataset.py (Normal class "
              "first) so the Normal-only baseline gets frozen properly.")
        stats = _fit_stats_from(feat_df)
        _save_stats(stats, stats_path)

    for col in FEATURE_COLUMNS:
        mean_val = stats[col]["mean"]
        std_val = stats[col]["std"] or 1e-6
        feat_df[f"{col}_norm"] = (feat_df[col] - mean_val) / std_val

    output_csv = os.path.join(output_dir, "clean_data.csv")
    feat_df.to_csv(output_csv, index=False)
    print(f"Saved {len(feat_df)} windowed feature vectors to: {output_csv}")

    return feat_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiBridge Data Preprocessing Pipeline")
    parser.add_argument("--input", type=str, default="sensor_data.csv", help="Path to input CSV")
    parser.add_argument("--output-dir", type=str, default="data_pipeline", help="Directory for preprocessed outputs")
    parser.add_argument("--sigma-shift", type=float, default=0.0, help="Optional N-Sigma mean shift experiment")
    parser.add_argument(
        "--fit-stats",
        action="store_true",
        help="Compute and freeze baseline stats from this run. Use ONLY on clean Normal-class data.",
    )
    args = parser.parse_args()

    preprocess_data(args.input, args.output_dir, args.sigma_shift, args.fit_stats)
