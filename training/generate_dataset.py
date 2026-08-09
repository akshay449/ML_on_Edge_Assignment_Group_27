"""
=========================================================
LogiBridge - 3-Class Training Dataset Generator
Module 4 - Task D1
=========================================================
Generates a balanced 3-Class dataset by running sensor simulations:
- Class 0 (Normal)   : ~20 mins simulation -- ALSO freezes training_stats.json
                        (Task C2: baseline stats must come from clean
                        Normal-class data only, and never be recomputed
                        from live/anomalous data afterwards)
- Class 1 (Warning)  : ~15 mins simulation (Temperature Drift)
- Class 2 (Critical) : ~30 mins simulation (Combined Anomalies) -- see note below

NOTE on the Critical-class duration deviating from the assignment's ~15-min /
~90-window guidance: the initial 15-minute run produced only 17 Critical-class
validation windows (20% split), meaning a single misclassification swings the
measured Class-2 recall by ~5.9 percentage points -- too coarse to reliably
certify the 95% safety threshold against, independent of how good the model
actually is. Doubling the duration roughly doubles both the model's Critical-
class training signal and the validation set's statistical precision (down to
~2.9 points per error). This is a deliberate, documented adaptation in
response to an observed recall gate failure, not an arbitrary change -- see
the Final Report's Section 4/5 writeup for the full investigation.
=========================================================
"""

import argparse
import os
import subprocess
import sys
import pandas as pd

# Dynamically set directory paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PIPELINE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data_pipeline"))

if DATA_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, DATA_PIPELINE_DIR)

from preprocessing import preprocess_data

SIMULATOR_PATH = os.path.join(DATA_PIPELINE_DIR, "simulator.py")
RAW_CSV = os.path.join(DATA_PIPELINE_DIR, "sensor_data.csv")
CLEAN_CSV = os.path.join(DATA_PIPELINE_DIR, "clean_data.csv")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "training_dataset.csv")


def chronological_split(df_clean, test_frac=0.2, gap_windows=3):
    """Splits one class's time-ordered windows into train/val WITHOUT shuffling.

    Sliding windows overlap: at WINDOW_SECONDS=30 / STEP_SECONDS=10, window N
    and N+1 share 67% of their raw samples, and N and N+2 still share 33%. A
    random shuffled split (sklearn's train_test_split, the previous approach)
    scatters these overlapping neighbours across the train/val boundary, so
    the "held-out" set contains windows that are substantially identical,
    sample-for-sample, to windows the model just trained on -- inflating
    reported accuracy (observed firsthand: accuracy jumped to 100% once the
    dataset grew, which is a data-leakage signature, not a model-quality one).

    This takes the LAST test_frac of each class's windows (in time order) as
    validation, and additionally drops `gap_windows` windows at the boundary
    so no validation window shares any raw sample with any training window.
    gap_windows=3 is enough for WINDOW/STEP=30/10 (overlap radius = 2); if
    those constants change, this gap should be recomputed as
    ceil(WINDOW_SECONDS / STEP_SECONDS).
    """
    n = len(df_clean)
    n_test = max(1, int(round(n * test_frac)))
    split_idx = n - n_test - gap_windows
    if split_idx < 1:
        # Class has too few windows for a full gap; shrink the gap rather
        # than the test set, and warn so it's visible rather than silent.
        print(f"[WARNING] Only {n} windows available -- shrinking train/val gap to "
              f"fit test_frac={test_frac}. Consider a longer simulation run for this class.")
        split_idx = max(1, n - n_test)
        gap_start = split_idx
    else:
        gap_start = split_idx

    train_df = df_clean.iloc[:split_idx].copy()
    val_df = df_clean.iloc[gap_start + gap_windows:].copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    return pd.concat([train_df, val_df], ignore_index=True)


def run_simulation_and_process(anomaly_mode, minutes, assigned_label, fit_stats=False):
    if os.path.exists(RAW_CSV):
        os.remove(RAW_CSV)

    print(f"\n---> Simulating '{anomaly_mode}' anomaly for {minutes} minutes...")

    # Run simulation using the active Python interpreter
    subprocess.run(
        [sys.executable, SIMULATOR_PATH, "--anomaly", anomaly_mode, "--minutes", str(minutes), "--fast"],
        check=True
    )

    # Process raw simulation CSV through the windowed preprocessing pipeline.
    # fit_stats=True is only ever passed for the Normal-class run below.
    preprocess_data(RAW_CSV, DATA_PIPELINE_DIR, fit_stats=fit_stats)

    # Load windowed feature dataset (still in time order at this point) and
    # set ground-truth target label
    df_clean = pd.read_csv(CLEAN_CSV)
    df_clean["label"] = assigned_label

    # Split THIS class's windows chronologically before anything gets
    # concatenated with other classes or shuffled -- see chronological_split().
    df_clean = chronological_split(df_clean)
    return df_clean


def main(normal_minutes=20, warning_minutes=15, critical_minutes=30):
    print("=" * 65)
    print("LogiBridge - 3-Class Training Dataset Generator")
    print(f"Durations: Normal={normal_minutes}min  Warning={warning_minutes}min  Critical={critical_minutes}min")
    print("=" * 65)

    # 1. Class 0: Normal Telemetry -- freezes training_stats.json
    df_normal = run_simulation_and_process("none", minutes=normal_minutes, assigned_label=0, fit_stats=True)

    # 2. Class 1: Warning / Temperature Drift -- reuses frozen stats
    df_warning = run_simulation_and_process("temp_drift", minutes=warning_minutes, assigned_label=1, fit_stats=False)

    # 3. Class 2: Critical / Combined Spoilage Anomalies -- reuses frozen stats
    df_critical = run_simulation_and_process("combined", minutes=critical_minutes, assigned_label=2, fit_stats=False)

    # Combine into single dataset
    final_df = pd.concat([df_normal, df_warning, df_critical], ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("Dataset Generation Complete!")
    print("Class Distribution (total):")
    print(final_df["label"].value_counts().rename({0: "Normal (0)", 1: "Warning (1)", 2: "Critical (2)"}))
    print("\nTrain/Val split (chronological, gapped -- no overlapping windows leak across it):")
    print(final_df.groupby(["label", "split"]).size().rename({0: "Normal", 1: "Warning", 2: "Critical"}))
    print(f"\nSaved final training dataset to: {OUTPUT_FILE}")
    print(f"Frozen baseline stats (Normal-class only): {os.path.join(DATA_PIPELINE_DIR, 'training_stats.json')}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiBridge 3-Class Training Dataset Generator")
    parser.add_argument("--normal-minutes", type=int, default=20, help="Normal-class simulation duration")
    parser.add_argument("--warning-minutes", type=int, default=15, help="Warning-class simulation duration")
    parser.add_argument("--critical-minutes", type=int, default=30,
                         help="Critical-class simulation duration (default doubled from the assignment's "
                              "~15min guidance -- see module docstring for why)")
    args = parser.parse_args()

    main(args.normal_minutes, args.warning_minutes, args.critical_minutes)
