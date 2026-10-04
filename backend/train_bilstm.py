"""
train_bilstm.py — SignTalk AI / Person A (ML Core), Prompt A1

Trains the BiLSTM gesture classifier on (30, 17, 2) keypoint-sequence .npy
files, splitting train/validation BY SIGNER (not random shuffle) to avoid
signer-identity leakage between splits.

Usage:
    python train_bilstm.py --data_dir data/sequences --labels_csv data/labels.csv \
        --output_dir runs/exp1 --epochs 100 --val_signers 0.2

Expects:
    labels.csv columns: filename, class, signer_id
    <data_dir>/<filename> : .npy file, shape (30, 17, 2)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
COORD_DIM = 2
FLAT_DIM = NUM_KEYPOINTS * COORD_DIM  # 34


def parse_args():
    p = argparse.ArgumentParser(description="Train SignTalk AI BiLSTM gesture classifier")
    p.add_argument("--data_dir", required=True, help="Folder containing .npy keypoint sequences")
    p.add_argument("--labels_csv", required=True, help="CSV with filename,class,signer_id")
    p.add_argument("--output_dir", required=True, help="Where to write model/plots/logs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_signers", type=float, default=0.2,
                    help="Fraction of unique signers (by ID, sorted) held out for validation")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(data_dir: str, labels_csv: str):
    df = pd.read_csv(labels_csv)
    required_cols = {"filename", "class", "signer_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing required columns: {missing}")

    sequences, labels, signers = [], [], []
    for _, row in df.iterrows():
        path = os.path.join(data_dir, row["filename"])
        if not os.path.exists(path):
            print(f"[warn] missing file, skipping: {path}")
            continue
        arr = np.load(path)
        if arr.shape != (SEQUENCE_LENGTH, NUM_KEYPOINTS, COORD_DIM):
            print(f"[warn] unexpected shape {arr.shape} for {path}, skipping")
            continue
        sequences.append(arr.reshape(SEQUENCE_LENGTH, FLAT_DIM))
        labels.append(str(row["class"]))
        signers.append(str(row["signer_id"]))

    if not sequences:
        raise RuntimeError("No valid sequences loaded — check data_dir/labels_csv.")

    X = np.stack(sequences).astype(np.float32)
    y_raw = np.array(labels)
    signer_ids = np.array(signers)
    return X, y_raw, signer_ids


def split_by_signer(signer_ids: np.ndarray, val_fraction: float, seed: int):
    unique_signers = sorted(set(signer_ids.tolist()))
    n_val = max(1, int(round(len(unique_signers) * val_fraction)))
    val_signers = set(unique_signers[-n_val:])  # hold out last 20% by default
    val_mask = np.array([s in val_signers for s in signer_ids])
    train_mask = ~val_mask
    print(f"Signers: {len(unique_signers)} total, {n_val} held out for validation: {sorted(val_signers)}")
    return train_mask, val_mask


def build_model(num_classes: int, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FLAT_DIM)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(max(6, len(class_names) * 0.5), max(5, len(class_names) * 0.5)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    X, y_raw, signer_ids = load_dataset(args.data_dir, args.labels_csv)

    class_names = sorted(set(y_raw.tolist()))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)

    train_mask, val_mask = split_by_signer(signer_ids, args.val_signers, args.seed)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Classes: {len(class_names)}")

    model = build_model(num_classes=len(class_names), lr=args.lr)
    model.summary()

    log_dir = os.path.join(args.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_path = os.path.join(args.output_dir, "best_model.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Reload best checkpoint before exporting/evaluating
    model = tf.keras.models.load_model(checkpoint_path)

    val_pred_probs = model.predict(X_val)
    val_pred = np.argmax(val_pred_probs, axis=1)
    final_val_acc = float(np.mean(val_pred == y_val))
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    plot_confusion_matrix(y_val, val_pred, class_names, os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_history(history, os.path.join(args.output_dir, "training_history.png"))

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump({str(i): c for i, c in enumerate(class_names)}, f, indent=2)

    # Export SavedModel
    saved_model_dir = os.path.join(args.output_dir, "saved_model")
    model.export(saved_model_dir)

    # Export float16-quantized TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    tflite_path = os.path.join(args.output_dir, "bilstm.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved: {checkpoint_path}, {saved_model_dir}, {tflite_path}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
