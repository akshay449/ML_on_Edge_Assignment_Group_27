"""
=========================================================
LogiBridge - Five-Metric Benchmarking Suite
Module 6 - Task F2
=========================================================
Benchmarks M1 (FP32), M2 (PTQ INT8), M3 (Pruned + PTQ INT8) on all five
required metrics: mean latency, p95 latency, model size, held-out
accuracy, and energy per inference (E = P * t, via psutil CPU% and a
TDP estimate). Produces benchmark_results.csv and the Pareto chart.
=========================================================
"""

import argparse
import csv
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf

DEFAULT_TDP_WATTS = 15.0  # dev-machine/laptop TDP estimate; pass --tdp 7.5 for
                           # Raspberry Pi 5 edge-hardware figures (see hardware_justification.md)


def _get_scale_zero_point(details):
    """Robustly extracts (scale, zero_point) from a TFLite tensor details dict.

    Modern TFLiteConverter output (the exact path this project uses --
    TFLiteConverter.from_keras_model + representative_dataset for INT8)
    populates the richer 'quantization_parameters' dict (scales/zero_points
    as arrays), NOT the legacy 'quantization' (scale, zero_point) tuple --
    that legacy field is often left as the placeholder (0.0, 0). Silently
    falling back to scale=1.0 when the legacy field reads 0.0 (the previous
    behaviour here) uses a WRONG scale rather than the real calibrated one,
    which explains INT8 models (M2/M3) producing wildly different --
    including implausibly perfect -- results compared to the FP32 model
    (M1) on the exact same weights.
    """
    qparams = details.get("quantization_parameters")
    if qparams and len(qparams.get("scales", [])) > 0:
        return float(qparams["scales"][0]), int(qparams["zero_points"][0])
    scale, zero_point = details.get("quantization", (0.0, 0))
    if not scale:
        raise ValueError(
            f"No valid quantization scale found for tensor '{details.get('name', '?')}' "
            f"(dtype={details.get('dtype')}). Both 'quantization_parameters' and legacy "
            f"'quantization' are empty -- this model may not actually be properly INT8 "
            f"calibrated. Re-run convert_ptq.py / prune_quantise.py and check for calibration warnings."
        )
    return float(scale), int(zero_point)


def _quantize_input(x_float, input_details):
    dtype = input_details["dtype"]
    if dtype in (np.int8, np.uint8):
        scale, zero_point = _get_scale_zero_point(input_details)
        q = x_float / scale + zero_point
        return np.clip(np.round(q), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return x_float.astype(np.float32)


def _dequantize_output(y_raw, output_details):
    dtype = output_details["dtype"]
    if dtype in (np.int8, np.uint8):
        scale, zero_point = _get_scale_zero_point(output_details)
        return (y_raw.astype(np.float32) - zero_point) * scale
    return y_raw.astype(np.float32)


def evaluate_accuracy(interpreter, input_details, output_details, X_val, y_val):
    """Returns (overall_accuracy_pct, per_class_recall_dict, confusion_matrix) or
    (None, None, None) if no validation data is available. per_class_recall_dict
    maps class label -> recall %, so Class-2 (Critical) recall -- the number that
    actually gates deployment per the project brief -- is visible directly,
    rather than inferred from overall accuracy arithmetic."""
    if X_val is None or y_val is None or len(X_val) == 0:
        return None, None, None

    y_true = []
    y_pred = []
    for i in range(len(X_val)):
        sample = np.expand_dims(X_val[i].astype(np.float32), axis=0)
        q_sample = _quantize_input(sample, input_details)
        interpreter.set_tensor(input_details["index"], q_sample)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_details["index"])
        out = _dequantize_output(raw_out, output_details)
        y_pred.append(int(np.argmax(out)))
        y_true.append(int(y_val[i]))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    overall_accuracy = 100.0 * float(np.mean(y_pred == y_true))

    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = {int(c): {int(p): 0 for p in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class_recall = {}
    for c in classes:
        total_c = sum(cm[c].values())
        correct_c = cm[c].get(c, 0)
        per_class_recall[c] = 100.0 * correct_c / total_c if total_c > 0 else None

    return overall_accuracy, per_class_recall, cm


def evaluate_tflite_model(model_path, num_runs=200, num_warmup=10, tdp_watts=DEFAULT_TDP_WATTS,
                           X_val=None, y_val=None, energy_window_s=0.5):
    if not os.path.exists(model_path):
        return None

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    is_int8 = input_details["dtype"] in (np.int8, np.uint8)
    size_kb = os.path.getsize(model_path) / 1024.0

    dummy = (
        np.ones(input_details["shape"], dtype=input_details["dtype"])
        if is_int8
        else np.ones(input_details["shape"], dtype=np.float32)
    )

    # Warmup (excluded from timing, per Task F2)
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details["index"], dummy)
        interpreter.invoke()

    # --- Latency: per-call timing over num_runs (unchanged) ---
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], dummy)
        interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000.0)  # ms
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # --- Energy: a single inference (~microseconds) completes far faster than
    # psutil's CPU-accounting granularity, so measuring cpu_percent() across one
    # inference (or even 200) reliably reads 0%. Instead, run a busy-wait
    # accumulation window of energy_window_s seconds -- doing as many inferences
    # as fit -- so there is a real, measurable amount of CPU-busy wall-clock time
    # for psutil to sample, then divide the window's total energy by however many
    # inferences it contained to get mJ/inference. ---
    psutil.cpu_percent(interval=None)  # discard stale reading
    accum_start = time.perf_counter()
    accum_count = 0
    while (time.perf_counter() - accum_start) < energy_window_s:
        interpreter.set_tensor(input_details["index"], dummy)
        interpreter.invoke()
        accum_count += 1
    elapsed_s = time.perf_counter() - accum_start
    cpu_percent = psutil.cpu_percent(interval=None)

    energy_note = "measured"
    if cpu_percent <= 0.0:
        # Still possible on a fast, mostly-idle multi-core machine if the OS
        # attributes the busy loop's load to a single core sampled at an
        # unlucky instant. Rather than silently reporting an impossible 0 mJ,
        # fall back to a documented worst-case single-core-saturated
        # assumption and flag it so the report doesn't misrepresent this as
        # a real measurement.
        cpu_percent = 100.0 / (psutil.cpu_count(logical=True) or 1)
        energy_note = "cpu_percent read 0%% even over a %.1fs busy window; " \
                       "reporting a floor-assumed single-core-saturated estimate instead" % energy_window_s

    avg_power_watts = tdp_watts * (cpu_percent / 100.0)
    total_energy_mj = avg_power_watts * elapsed_s * 1000.0
    energy_per_run_mj = total_energy_mj / accum_count if accum_count else 0.0

    accuracy, per_class_recall, confusion = evaluate_accuracy(interpreter, input_details, output_details, X_val, y_val)
    class2_recall = per_class_recall.get(2) if per_class_recall else None

    return {
        "model": os.path.basename(model_path),
        "size_kb": size_kb,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "accuracy_pct": accuracy,
        "class2_recall_pct": class2_recall,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion,
        "energy_mj": energy_per_run_mj,
        "cpu_percent": cpu_percent,
        "energy_note": energy_note,
        "energy_accum_runs": accum_count,
    }


def _load_validation_split(models_dir):
    val_path = os.path.join(models_dir, "validation_split.npz")
    if not os.path.exists(val_path):
        print(f"[WARNING] {val_path} not found -- accuracy column will be blank. "
              f"Run train_model.py first (it now saves this file).")
        return None, None
    data = np.load(val_path)
    return data["X_test"], data["y_test"]


def run_benchmark(models_dir="models", results_dir="optimisation/results", tdp_watts=DEFAULT_TDP_WATTS,
                   energy_window_s=0.5):
    os.makedirs(results_dir, exist_ok=True)

    # Exact filenames produced by train_model.py (M1), convert_ptq.py (M2),
    # and prune_quantise.py (M3) -- also the Dockerfile's default MODEL_PATH.
    models = [
        os.path.join(models_dir, "cold_chain_model.tflite"),
        os.path.join(models_dir, "logibridge_int8.tflite"),
        os.path.join(models_dir, "logibridge_pruned_int8.tflite"),
    ]

    X_val, y_val = _load_validation_split(models_dir)

    results = []
    header = f"{'Model':<32} | {'Size(KB)':<9} | {'Avg(ms)':<8} | {'P95(ms)':<8} | {'Acc(%)':<7} | {'C2 Recall(%)':<13} | {'Energy(mJ)':<10}"
    print(header)
    print("-" * len(header))

    for m in models:
        res = evaluate_tflite_model(m, tdp_watts=tdp_watts, X_val=X_val, y_val=y_val, energy_window_s=energy_window_s)
        if res:
            results.append(res)
            acc_str = f"{res['accuracy_pct']:.2f}" if res["accuracy_pct"] is not None else "N/A"
            c2_str = f"{res['class2_recall_pct']:.2f}" if res["class2_recall_pct"] is not None else "N/A"
            print(
                f"{res['model']:<32} | {res['size_kb']:<9.2f} | {res['avg_latency_ms']:<8.4f} | "
                f"{res['p95_latency_ms']:<8.4f} | {acc_str:<7} | {c2_str:<13} | {res['energy_mj']:<10.4f}"
            )
            if res["accuracy_pct"] is not None and res["accuracy_pct"] < 88.0:
                print(f"  [GATE FAIL] {res['model']}: accuracy {res['accuracy_pct']:.2f}% < 88% required threshold")
            if res["class2_recall_pct"] is not None and res["class2_recall_pct"] < 95.0:
                print(f"  [GATE FAIL] {res['model']}: Class-2 recall {res['class2_recall_pct']:.2f}% < 95% required threshold")
            if res["per_class_recall"] is not None:
                print(f"  Per-class recall: {res['per_class_recall']}")
            if res["confusion_matrix"] is not None:
                print(f"  Confusion matrix (rows=true, cols=predicted): {res['confusion_matrix']}")
            if res["energy_note"] != "measured":
                print(f"  [NOTE] {res['model']}: {res['energy_note']} "
                      f"({res['energy_accum_runs']} inferences accumulated over the window)")
        else:
            print(f"[SKIPPED] {m} not found")

    if not results:
        print("\n[!] No models found to benchmark. Run train_model.py, convert_ptq.py, "
              "and prune_quantise.py first.")
        return

    # CSV keeps only flat/scalar columns; confusion_matrix and per_class_recall
    # are dicts and go to the companion JSON file instead so the CSV stays a
    # clean, spreadsheet-friendly table.
    csv_fieldnames = ["model", "size_kb", "avg_latency_ms", "p95_latency_ms",
                       "accuracy_pct", "class2_recall_pct", "energy_mj", "cpu_percent",
                       "energy_note", "energy_accum_runs"]
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[+] Benchmark results saved to {csv_path}")

    json_path = os.path.join(results_dir, "benchmark_results_detailed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Full detail (incl. per-class recall + confusion matrices) saved to {json_path}")

    sizes = [r["size_kb"] for r in results]
    lats = [r["avg_latency_ms"] for r in results]
    labels = [r["model"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.scatter(sizes, lats, color="blue", s=100)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (sizes[i], lats[i]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Model Size (KB)")
    plt.ylabel("Inference Latency (ms)")
    plt.title("Pareto Optimization Frontier (Size vs Latency)")
    plt.grid(True)
    pareto_path = os.path.join(results_dir, "pareto_chart.png")
    plt.savefig(pareto_path)
    print(f"[+] Pareto frontier plot saved to {pareto_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiBridge Model Benchmarking Suite")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="optimisation/results")
    parser.add_argument("--tdp", type=float, default=DEFAULT_TDP_WATTS,
                         help="TDP watts estimate for energy calc (use 7.5 for Pi 5 figures)")
    parser.add_argument("--energy-window", type=float, default=0.5,
                         help="Seconds to busy-loop inferences for the energy measurement "
                              "(default 0.5s -- a single inference is too fast for psutil to "
                              "register non-zero CPU% over)")
    args = parser.parse_args()

    run_benchmark(args.models_dir, args.results_dir, args.tdp, args.energy_window)
