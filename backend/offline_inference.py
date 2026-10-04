"""
offline_inference.py — SignTalk AI / Person A (ML Core), Prompt A2

Fully offline, on-device inference pipeline: MoveNet.tflite -> keypoint
normalization/smoothing (reused from keypoint_utils.py) -> BiLSTM.tflite
sequence classification. Zero network access required at runtime.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

from keypoint_utils import (
    SequenceBuffer,
    TemporalSmoother,
    normalize_keypoints,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MOVENET_TFLITE = os.path.join(MODELS_DIR, "movenet.tflite")
DEFAULT_BILSTM_TFLITE = os.path.join(MODELS_DIR, "bilstm.tflite")
DEFAULT_LABELS_JSON = os.path.join(MODELS_DIR, "labels.json")

MOVENET_INPUT_SIZE = 256
SEQUENCE_LENGTH = 30
SMOOTHING_WINDOW = 4
STABILIZATION_AGREEMENT_COUNT = 3
COOLDOWN_SECONDS = 1.5


class OfflineGesturePipeline:
    """
    Zero-network gesture recognition pipeline for on-device / airplane-mode
    demo use. Mirrors main.py's ConnectionState pipeline but runs both
    models via tf.lite.Interpreter instead of TF Hub / a Keras SavedModel.
    """

    def __init__(
        self,
        movenet_path: str = DEFAULT_MOVENET_TFLITE,
        bilstm_path: str = DEFAULT_BILSTM_TFLITE,
        labels_path: str = DEFAULT_LABELS_JSON,
    ):
        if not os.path.exists(movenet_path):
            raise FileNotFoundError(
                f"MoveNet TFLite model not found at {movenet_path}. "
                f"Run: python convert_to_tflite.py --download_movenet"
            )
        if not os.path.exists(bilstm_path):
            raise FileNotFoundError(
                f"BiLSTM TFLite model not found at {bilstm_path}. "
                f"Run train_bilstm.py then convert_to_tflite.py."
            )

        self.movenet_interpreter = tf.lite.Interpreter(model_path=movenet_path)
        self.movenet_interpreter.allocate_tensors()
        self._movenet_input = self.movenet_interpreter.get_input_details()[0]
        self._movenet_output = self.movenet_interpreter.get_output_details()[0]

        self.bilstm_interpreter = tf.lite.Interpreter(model_path=bilstm_path)
        self.bilstm_interpreter.allocate_tensors()
        self._bilstm_input = self.bilstm_interpreter.get_input_details()[0]
        self._bilstm_output = self.bilstm_interpreter.get_output_details()[0]

        self.labels: dict[str, str] = {}
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self.labels = json.load(f)

        self.smoother = TemporalSmoother(window=SMOOTHING_WINDOW)
        self.buffer = SequenceBuffer(maxlen=SEQUENCE_LENGTH)
        self._recent_predictions: list[str] = []
        self._last_emitted_label: Optional[str] = None
        self._last_emitted_at: float = 0.0

    # -- MoveNet (TFLite) -------------------------------------------------
    def _extract_keypoints_tflite(self, frame_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = tf.image.resize_with_pad(
            np.expand_dims(image_rgb, axis=0), MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE
        )
        input_dtype = self._movenet_input["dtype"]
        img = tf.cast(img, dtype=input_dtype).numpy()

        self.movenet_interpreter.set_tensor(self._movenet_input["index"], img)
        self.movenet_interpreter.invoke()
        keypoints = self.movenet_interpreter.get_tensor(self._movenet_output["index"])
        return keypoints[0, 0, :, :]  # (17, 3)

    # -- BiLSTM (TFLite) ----------------------------------------------------
    def _classify_sequence_tflite(self, sequence: np.ndarray) -> dict:
        flat = sequence.reshape(1, SEQUENCE_LENGTH, 34).astype(self._bilstm_input["dtype"])
        self.bilstm_interpreter.set_tensor(self._bilstm_input["index"], flat)
        self.bilstm_interpreter.invoke()
        probs = self.bilstm_interpreter.get_tensor(self._bilstm_output["index"])[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        label = self.labels.get(str(class_idx), f"class_{class_idx}")
        return {"label": label, "confidence": confidence}

    # -- Public API ---------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """Runs one frame through the full offline pipeline. Returns a
        stabilized emission dict or None, identical contract to main.py's
        ConnectionState.process_frame()."""
        raw_keypoints = self._extract_keypoints_tflite(frame_bgr)
        normalized = normalize_keypoints(raw_keypoints)
        smoothed = self.smoother.smooth(normalized)
        self.buffer.push(smoothed)

        sequence = self.buffer.get_sequence()
        if sequence is None:
            return None

        result = self._classify_sequence_tflite(sequence)
        return self._stabilize(result)

    def _stabilize(self, result: dict) -> Optional[dict]:
        label, confidence = result["label"], result["confidence"]
        now = time.time()

        self._recent_predictions.append(label)
        self._recent_predictions = self._recent_predictions[-STABILIZATION_AGREEMENT_COUNT:]

        agrees = (
            len(self._recent_predictions) == STABILIZATION_AGREEMENT_COUNT
            and len(set(self._recent_predictions)) == 1
        )
        if not agrees:
            return None

        if label == self._last_emitted_label and (now - self._last_emitted_at) < COOLDOWN_SECONDS:
            return None

        self._last_emitted_label = label
        self._last_emitted_at = now
        return {"label": label, "confidence": confidence, "timestamp": now}

    def reset(self) -> None:
        self.smoother.reset()
        self.buffer.reset()
        self._recent_predictions = []


if __name__ == "__main__":
    # Minimal smoke test using a webcam, if available.
    pipeline = OfflineGesturePipeline()
    cap = cv2.VideoCapture(0)
    print("Running offline pipeline on webcam. Press 'q' to quit.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        emission = pipeline.process_frame(frame)
        if emission:
            print(emission)
        cv2.imshow("SignTalk AI — Offline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
