
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

from classify import add_velocity_features  # shared with inference — see classify.py

SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
COORD_DIM = 2
FLAT_DIM = NUM_KEYPOINTS * COORD_DIM  # 34

# Augmentation magnitudes — tuned for the shoulder-centered, scale-normalized
# (17, 2) representation from keypoint_utils.normalize_keypoints(). No
# left-right mirroring: the live pipeline deliberately never mirrors frames
# (see README "Known gotchas"), and mirroring would also flip
# handedness-dependent signs, so it's not a safe augmentation here.
def parse_args():
    p = argparse.ArgumentParser(description="Train SignTalk AI BiLSTM gesture classifier")
    p.add_argument("--data_dir", required=True, help="Folder containing .npy keypoint sequences")
    p.add_argument("--labels_csv", required=True, help="CSV with filename,class,signer_id")
    p.add_argument("--output_dir", required=True, help="Where to write model/plots/logs")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_signers", type=float, default=0.2,
                    help="Fraction of unique signers (by ID, sorted) held out for validation")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", dest="augment", action="store_true", default=False,
                    help="Apply on-the-fly rotation/scale/jitter/time-warp augmentation to training data")
    p.add_argument("--no_augment", dest="augment", action="store_false",
                    help="Disable augmentation (default)")
    # Augmentation magnitudes — tuned for the shoulder-centered, scale-normalized
    # (17, 2) representation from keypoint_utils.normalize_keypoints(). No
    # left-right mirroring flag: the live pipeline deliberately never mirrors
    # frames (see README "Known gotchas"), and mirroring would also flip
    # handedness-dependent signs, so it's not offered here.
    p.add_argument("--aug_rotation_deg", type=float, default=15.0)
    p.add_argument("--aug_scale_jitter", type=float, default=0.10)
    p.add_argument("--aug_shift_jitter", type=float, default=0.05)
    p.add_argument("--aug_noise_std", type=float, default=0.02)
    p.add_argument("--aug_time_warp_frac", type=float, default=0.2,
                    help="Resample to a random length in [1-frac, 1+frac] * 30, then back to 30. 0 disables.")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--l2", type=float, default=0.0, help="L2 weight regularization strength")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=20, help="Early-stopping patience (epochs)")
    p.add_argument("--split_mode", choices=["signer", "stratified"], default="signer",
                    help="signer: hold out by signer_id (correct only with real signer metadata). "
                         "stratified: per-class random split, every class represented in val — use "
                         "this when signer_id is a per_video/per_session placeholder, not real identity.")
    p.add_argument("--use_velocity", action="store_true", default=False,
                    help="Append frame-to-frame velocity (dx, dy per keypoint) to the raw (x, y) "
                         "position features, giving the model motion instead of position alone — "
                         "the pose pipeline otherwise only provides wrist position per frame. "
                         "Sets SIGNTALK_USE_VELOCITY_FEATURES=true on the deployed model's env to "
                         "match at inference (see classify.py's add_velocity_features).")
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
        sequences.append(arr)
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


def split_stratified(y: np.ndarray, val_fraction: float, seed: int):
    """
    Per-class random split — every class gets val_fraction of its own
    examples held out, so every class is actually represented in
    validation. Signer-based splitting only prevents identity leakage when
    signer_id is real; preprocess_include.py's default `per_video` mode
    makes every clip its own fake "signer", so split_by_signer just adds
    arbitrary alphabetical-split noise without the leakage protection it's
    meant to provide — with ~13-22 examples spread over 40 classes, that
    noise means many classes end up with 2-3 (or zero) validation examples,
    making per-class accuracy numbers meaningless. Only use this when
    signer_id is a real per_video/per_session placeholder, not genuine
    signer metadata (in which case split_by_signer is the correct choice).
    """
    rng = np.random.RandomState(seed)
    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    for cls in sorted(set(y.tolist())):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_mask[idx[:n_val]] = True
        train_mask[idx[n_val:]] = True
    print(f"Stratified split: {train_mask.sum()} train, {val_mask.sum()} val, "
          f"every one of {len(set(y.tolist()))} classes represented in both")
    return train_mask, val_mask


def augment_sequence(seq: tf.Tensor, rotation_deg: float, scale_jitter: float,
                      shift_jitter: float, noise_std: float, time_warp_frac: float) -> tf.Tensor:
    """
    Random rotation + scale + translation + gaussian jitter + temporal
    speed-warp on one (SEQUENCE_LENGTH, NUM_KEYPOINTS, COORD_DIM) sequence.
    Applied per-example (not batched) so each sample in a batch gets an
    independent random transform. Operates entirely in normalized
    keypoint-coordinate space — no raw video/image involved.
    """
    theta = tf.random.uniform([], -rotation_deg, rotation_deg) * (np.pi / 180.0)
    cos_t, sin_t = tf.cos(theta), tf.sin(theta)
    rot = tf.stack([[cos_t, -sin_t], [sin_t, cos_t]])
    flat_xy = tf.reshape(seq, [-1, COORD_DIM])
    seq = tf.reshape(tf.matmul(flat_xy, rot, transpose_b=True), tf.shape(seq))

    scale = tf.random.uniform([], 1.0 - scale_jitter, 1.0 + scale_jitter)
    seq = seq * scale

    shift = tf.random.uniform([COORD_DIM], -shift_jitter, shift_jitter)
    seq = seq + shift

    seq = seq + tf.random.normal(tf.shape(seq), stddev=noise_std)

    if time_warp_frac > 0:
        min_len = int(round(SEQUENCE_LENGTH * (1.0 - time_warp_frac)))
        max_len = int(round(SEQUENCE_LENGTH * (1.0 + time_warp_frac)))
        new_len = tf.random.uniform([], min_len, max_len + 1, dtype=tf.int32)
        warped = tf.image.resize(tf.reshape(seq, [1, SEQUENCE_LENGTH, FLAT_DIM, 1]), [new_len, FLAT_DIM])
        seq = tf.reshape(
            tf.image.resize(warped, [SEQUENCE_LENGTH, FLAT_DIM]),
            [SEQUENCE_LENGTH, NUM_KEYPOINTS, COORD_DIM],
        )
    return seq


def _add_velocity_tf(seq):
    """TF-graph equivalent of classify.add_velocity_features(), for use
    inside a tf.data pipeline (after augmentation, which expects raw
    (17, 2) position tensors — velocity is computed on the possibly-
    augmented sequence, matching what a live augmented-then-classified
    sequence would look like)."""
    velocity = seq[1:] - seq[:-1]
    velocity = tf.concat([tf.zeros_like(seq[:1]), velocity], axis=0)
    return tf.concat([seq, velocity], axis=-1)


def make_dataset(X, y, num_classes, batch_size, training, augment, seed, use_velocity, flat_dim, aug_kwargs=None):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
        if augment:
            ds = ds.map(lambda seq, label: (augment_sequence(seq, **aug_kwargs), label),
                        num_parallel_calls=tf.data.AUTOTUNE)
    if use_velocity:
        ds = ds.map(lambda seq, label: (_add_velocity_tf(seq), label), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda seq, label: (tf.reshape(seq, [SEQUENCE_LENGTH, flat_dim]), tf.one_hot(label, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int, lr: float, dropout: float, l2_reg: float,
                 label_smoothing: float, flat_dim: int) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, flat_dim)),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, kernel_regularizer=reg, recurrent_regularizer=reg)
        ),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
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

    if args.split_mode == "stratified":
        train_mask, val_mask = split_stratified(y, args.val_signers, args.seed)
    else:
        train_mask, val_mask = split_by_signer(signer_ids, args.val_signers, args.seed)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Classes: {len(class_names)}")
    print(f"Augmentation: {'on' if args.augment else 'off'}, dropout={args.dropout}, "
          f"l2={args.l2}, label_smoothing={args.label_smoothing}, use_velocity={args.use_velocity}")

    aug_kwargs = dict(
        rotation_deg=args.aug_rotation_deg,
        scale_jitter=args.aug_scale_jitter,
        shift_jitter=args.aug_shift_jitter,
        noise_std=args.aug_noise_std,
        time_warp_frac=args.aug_time_warp_frac,
    )
    if args.augment:
        print(f"Augmentation params: {aug_kwargs}")

    flat_dim = FLAT_DIM * 2 if args.use_velocity else FLAT_DIM
    num_classes = len(class_names)
    train_ds = make_dataset(X_train, y_train, num_classes, args.batch_size,
                             training=True, augment=args.augment, seed=args.seed,
                             use_velocity=args.use_velocity, flat_dim=flat_dim, aug_kwargs=aug_kwargs)
    val_ds = make_dataset(X_val, y_val, num_classes, args.batch_size,
                           training=False, augment=False, seed=args.seed,
                           use_velocity=args.use_velocity, flat_dim=flat_dim)

    model = build_model(num_classes=num_classes, lr=args.lr, dropout=args.dropout,
                         l2_reg=args.l2, label_smoothing=args.label_smoothing, flat_dim=flat_dim)
    model.summary()

    log_dir = os.path.join(args.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_path = os.path.join(args.output_dir, "best_model.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Reload best checkpoint before exporting/evaluating
    model = tf.keras.models.load_model(checkpoint_path)

    X_val_for_eval = np.stack([add_velocity_features(s) for s in X_val]) if args.use_velocity else X_val
    X_val_flat = X_val_for_eval.reshape(len(X_val), SEQUENCE_LENGTH, flat_dim)
    val_pred_probs = model.predict(X_val_flat)
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

    # Export float16-quantized TFLite. The Bidirectional LSTM's backward
    # pass needs Select TF ops (Flex) support to convert at all — plain
    # TFLITE_BUILTINS fails with "tf.TensorListReserve op requires
    # element_shape to be static". The exported model requires the Flex
    # delegate at inference time as a result (see tensorflow.org/lite/guide/ops_select).
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    tflite_path = os.path.join(args.output_dir, "bilstm.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved: {checkpoint_path}, {saved_model_dir}, {tflite_path}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
