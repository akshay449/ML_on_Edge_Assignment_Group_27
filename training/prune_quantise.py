"""
=========================================================
LogiBridge - Pruning & Quantization Pipeline (M3)
Module 4 - Task D2
=========================================================
Applies Magnitude-based Pruning (35% sparsity) and INT8
Quantization to build the optimized edge model (M3).
=========================================================
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "data_pipeline")
if DATA_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, DATA_PIPELINE_DIR)

from preprocessing import FEATURE_COLUMNS

NORM_FEATURE_COLUMNS = [f"{c}_norm" for c in FEATURE_COLUMNS]


def locate_file(filename):
    """Finds target file across root, script folder, and relative paths."""
    possible_paths = [
        filename,
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(SCRIPT_DIR, filename),
        os.path.join("..", filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not locate required file: '{filename}'")


def representative_dataset_gen():
    """Generates calibration samples for INT8 quantization scale/zero-point parameters.

    Uses training_dataset.csv (the full 3-class combined dataset from
    generate_dataset.py), TRAIN split only -- not clean_data.csv, which only
    ever contains whichever single class generate_dataset.py processed last
    (Critical, in the current pipeline order). PTQ calibration should also see
    the full input distribution the model will actually face, not one class.
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

    for i in range(min(200, len(X))):
        yield [np.expand_dims(X[i], axis=0)]


def apply_pruning_and_quantization(data_path="training/training_dataset.csv", output_dir="models",
                                    base_weights_path="models/baseline_weights.npz",
                                    target_sparsity=0.20, fine_tune_epochs=25):
    resolved_data_path = locate_file(data_path)
    print(f"Loading preprocessed dataset from: {resolved_data_path}")
    df = pd.read_csv(resolved_data_path)

    missing = [c for c in NORM_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{resolved_data_path} is missing expected columns {missing}. "
            f"Re-run generate_dataset.py with the current preprocessing.py."
        )

    if "split" not in df.columns:
        raise ValueError(
            f"{resolved_data_path} has no 'split' column -- this looks like an older "
            f"clean_data.csv (single-class) rather than the full training_dataset.csv "
            f"generate_dataset.py now produces. Fine-tuning on a single class causes "
            f"catastrophic forgetting of the others. Re-run generate_dataset.py, then "
            f"point --data at ../training/training_dataset.csv."
        )

    # FINE-TUNE ON TRAIN SPLIT ONLY, ACROSS ALL 3 CLASSES. Fine-tuning on
    # clean_data.csv (the previous default) trained exclusively on whichever
    # class generate_dataset.py processed last (Critical) -- every label the
    # fine-tune ever saw was "2", which pushes the model toward always
    # predicting Critical regardless of input. That, not pruning sparsity,
    # is what caused the Warning-class collapse seen across every sparsity
    # level tested (50%/35%/20%) in earlier runs.
    train_df = df[df["split"] == "train"]
    print(f"Fine-tuning on {len(train_df)} TRAIN-split windows across all 3 classes "
          f"(label distribution: {train_df['label'].value_counts().to_dict()})")

    X = train_df[NORM_FEATURE_COLUMNS].values
    y = train_df["label"].values

    # Rebuild the EXACT M1 architecture (train_model.py) and load its trained
    # Rebuild the EXACT M1 architecture (train_model.py) and load its trained
    # weights as raw NumPy arrays via set_weights(), NOT through any Keras
    # file format. Reason: this script imports tensorflow_model_optimization,
    # which forces the legacy tf_keras backend internally. Both
    # tf.keras.models.load_model() (full model) and model.load_weights()
    # (HDF5 weights-only) route through Keras's own version-specific
    # serializers, and native-Keras-3-written files can fail to load under
    # the legacy backend either way -- a "batch_shape" InputLayer error for
    # load_model(), or an "expected N variables, received 0" error for
    # load_weights(), depending on which serializer wrote the file.
    # get_weights()/set_weights() are plain lists of NumPy arrays with no
    # version-specific format involved, so they sidestep this class of bug
    # entirely rather than just moving it to a different file format.
    print(f"Rebuilding M1 architecture and loading trained weight arrays from: {base_weights_path}")
    try:
        resolved_weights_path = locate_file(base_weights_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find '{base_weights_path}'. This file is written by the current "
            f"version of train_model.py -- if you trained M1 before this fix was applied, "
            f"re-run train_model.py once (it's quick) so baseline_weights.npz exists, then "
            f"re-run this script."
        )
    base_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    with np.load(resolved_weights_path) as npz:
        n_arrays = len(npz.files)
        weight_arrays = [npz[f"w{i}"] for i in range(n_arrays)]
    base_model.set_weights(weight_arrays)
    print(f"Loaded {n_arrays} weight arrays into the rebuilt architecture.")

    # Apply Magnitude-based Pruning.
    # NOTE -- experiment history, kept here since it's directly relevant to the
    # Final Report's reflection section:
    #   50% sparsity, 10 epochs  -> 40.68% accuracy (exact majority-class
    #     fraction; later traced to a SEPARATE weight-loading bug, now fixed).
    #   35% sparsity, 10 epochs  -> 71.19% accuracy, Warning recall 11.1%
    #     (16/18 Warning samples misclassified as Critical).
    #   35% sparsity, 25 epochs  -> 69.49% accuracy, Warning recall 0.0%
    #     (ALL Warning samples misclassified as Critical). More fine-tuning
    #     time made the Warning/Critical collapse MORE complete, not less --
    #     which rules out "just needs more epochs to converge" and points at
    #     35% sparsity removing capacity this specific 32-16-3 MLP needs to
    #     hold that particular boundary at all.
    #   20% sparsity, 25 epochs  -> testing now: does less pruning preserve
    #     the Warning/Critical boundary? If this ALSO collapses, the honest
    #     conclusion is this architecture doesn't have spare capacity to
    #     prune at any meaningful level, and M2 (unpruned INT8) is the
    #     correct deployment recommendation -- see Final Report Section 4.
    TARGET_SPARSITY = target_sparsity
    FINE_TUNE_EPOCHS = fine_tune_epochs
    steps_per_epoch = np.ceil(len(X) / 16).astype(int)
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=TARGET_SPARSITY,
            begin_step=0,
            end_step=steps_per_epoch * FINE_TUNE_EPOCHS
        )
    }

    model_for_pruning = prune_low_magnitude(base_model, **pruning_params)
    # Lower learning rate than M1's original training run: this is a
    # fine-tune of an already-converged model, not training from scratch,
    # so a smaller step size avoids undoing the pretrained weights before
    # the pruning schedule has finished ramping up sparsity.
    model_for_pruning.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"Fine-tuning the pretrained M1 baseline with {TARGET_SPARSITY*100:.0f}% target "
          f"sparsity pruning over {FINE_TUNE_EPOCHS} epochs...")
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    model_for_pruning.fit(X, y, epochs=FINE_TUNE_EPOCHS, batch_size=16, callbacks=callbacks, verbose=1)

    # Strip pruning wrappers for export
    stripped_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    # Convert stripped model to INT8 Quantized TFLite (M3)
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting pruned model to INT8 TFLite (M3)...")
    tflite_pruned_quant_model = converter.convert()

    # NOTE: output_dir is resolved relative to the current working directory,
    # exactly like train_model.py and convert_ptq.py do -- it must NOT be
    # re-prefixed with PROJECT_ROOT here, or a call like
    # `--output-dir ../models` run from training/ ends up one directory
    # ABOVE the project root instead of at logibridge/models/.
    resolved_output_dir = output_dir
    os.makedirs(resolved_output_dir, exist_ok=True)

    # Save artifact under the name inference_service.py / benchmark.py /
    # the Dockerfile's default MODEL_PATH all expect for the M3 variant.
    output_path = os.path.join(resolved_output_dir, "logibridge_pruned_int8.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_pruned_quant_model)

    print(f"Successfully saved Pruned + Quantized TFLite model (M3) to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LogiBridge Pruning & Quantization Pipeline")
    parser.add_argument("--data", type=str, default="training/training_dataset.csv",
                         help="Path to training_dataset.csv (full 3-class dataset -- NOT clean_data.csv, "
                              "which only ever holds the last class generate_dataset.py processed)")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save output models")
    parser.add_argument("--base-weights", type=str, default="models/baseline_weights.npz",
                         help="Path to M1's trained weight arrays (written by train_model.py; run that first)")
    parser.add_argument("--sparsity", type=float, default=0.20,
                         help="Target pruning sparsity, 0-1 (spec calls for 0.35; 0.20 was being tested "
                              "when the real bug turned out to be single-class fine-tuning data, now fixed -- "
                              "worth re-testing 0.35 now that this is corrected)")
    parser.add_argument("--epochs", type=int, default=25, help="Fine-tune epochs under the pruning schedule")
    args = parser.parse_args()

    apply_pruning_and_quantization(args.data, args.output_dir, args.base_weights, args.sparsity, args.epochs)
