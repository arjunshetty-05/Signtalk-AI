# Technology Stack

Two stacks are described: the **target (v3)** from `PROJECT_CONTEXT.md`, and the
**current (v2)** stack actually present in the repo (from `requirements.txt`,
`package.json`, and the code). v3 deliberately simplifies and swaps several v2
choices.

---

## Target stack (v3)

| Subsystem | Choice | Notes |
|---|---|---|
| Language / runtime | Python 3.10 / 3.11 | type hints, ruff + black, pytest (markers: `unit`, `acceptance`) |
| Landmark extraction | **MediaPipe Tasks Holistic** (or Pose + Hand landmarkers) | 21 landmarks per hand + upper-body pose, 2D only (depth hurt in studies); legacy `mp.solutions.*` is unsupported |
| ML framework | **PyTorch** (single framework) | export **ONNX** for serving, parity-tested against PyTorch |
| Models | Ensemble: M1 pose-TCN/BiGRU, M2 small Transformer, M3 optional RGB-crop video | pretrained init from OpenHands/INCLUDE where usable |
| Calibration / fusion | Temperature scaling + weighted average | weights tuned on validation (grid search) |
| Decision | Accept / Confirm / Reject engine | thresholds from the risk-coverage curve |
| Backend | **FastAPI** (thin routers), Pydantic validation, rate limiting, CORS, `/metrics` | all ML logic in `signtalk_core` package |
| Storage | **SQLite** (default) | Firestore optional behind a Storage interface |
| Auth | **Guest mode** default | Firebase optional by config; short-lived ticket for WS, never a JWT in the URL |
| Sentence LLM | Gemini via `GEMINI_MODEL` env var | provider interface (gemini / none / ollama); 4 s timeout; strict-JSON + safety check |
| Translation | Scripted table + LLM JSON (en/hi/kn) | SQLite cache; phrasebook offline; Google Cloud Translate optional |
| Text-to-speech | **Browser `SpeechSynthesis`** by default | optional gTTS via `/api/tts`; **Coqui dropped** |
| Speech-to-text | faster-whisper / openai-whisper (small/base) + VAD | browser Web Speech API fallback |
| Emotion | DeepFace (optional, low priority) | default `neutral`; only nudges sentence tone |
| Frontend | React + Vite + Tailwind | preflight, capture, enroll, sentence UI |
| Serving format | ONNX | portable, fast inference |

### Target `.env` keys

```
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash       # [VERIFY]; Google retires models often
LLM_PROVIDER=gemini                 # gemini | none | ollama
LLM_TIMEOUT_SECONDS=4
WHISPER_MODEL=small                 # small | base | tiny
ENABLE_EMOTION=false
STORAGE_BACKEND=sqlite              # sqlite | firestore
AUTH_MODE=guest                     # guest | firebase
FIREBASE_SERVICE_ACCOUNT_PATH=
CORS_ORIGINS=http://localhost:3000
USE_GPU=auto
```

### Target config (`config/signtalk.yaml`, placeholder values)

```yaml
sequence_length: 32          # pose models
rgb_frames: 16
smoothing_window: 3
quality: {min_hands_visible_pct: 0.7, min_seconds: 0.4, max_seconds: 5.0}
tta: {views: 5, trim_frames: [0, 3, 6]}
decision:
  t_accept: 0.85
  m_accept: 0.30
  min_model_agreement: all
  tta_agreement: 0.8
  t_confirm: 0.40
sentence: {idle_seconds: 2.5, max_words: 8, mode: strict}
context_prior: {enabled: false, scenario: null}
```

> The decision thresholds above are placeholders; `tools/pick_thresholds.py`
> replaces them with values computed from validation data.

---

## Current stack (v2 reference, in the repo today)

### Backend (`backend/requirements.txt`, pinned)

| Subsystem | Technology |
|---|---|
| Web framework / server | FastAPI 0.111.0, Uvicorn 0.30.1, python-multipart, Pydantic 2.7.4 |
| Auth / security / rate limiting | firebase-admin 6.5.0, slowapi 0.1.9 |
| Realtime | python-socketio 5.11.3 |
| Pose / keypoints | TensorFlow 2.15.0 + tensorflow-hub 0.16.1 (**MoveNet Thunder, 17 keypoints**), OpenCV 4.10 |
| Sequence model | TensorFlow / Keras **BiLSTM** (128 units, dropout 0.3); TFLite float16 for offline |
| Emotion | DeepFace 0.0.93 (7 classes) |
| NLP correction | google-generativeai 0.7.1 (**Gemini 2.0 Flash**) / transformers 4.42.3 + torch 2.3.1 (Flan-T5 offline) |
| STT | openai-whisper 20231117 (Small) |
| TTS | gTTS 2.5.1 (online); Coqui TTS offline (separate venv, `requirements-offline-tts.txt`) |
| Translation | google-cloud-translate 3.15.3; in-memory LRU → Firestore → `phrasebook.json` |
| Data / science | numpy 1.24.3, pandas 1.5.3, scikit-learn 1.3.2, matplotlib, seaborn |

Notable pins / gotchas baked into `requirements.txt`:
- `setuptools<81` must install before tensorflow-hub (newer setuptools dropped
  the bundled `pkg_resources` that tensorflow-hub imports).
- TensorFlow pinned to 2.15.0 (not 2.13.1) to avoid a `typing-extensions<4.6.0`
  cap that conflicts with FastAPI/Pydantic/torch.
- Coqui TTS is deliberately **not** in the main requirements — its pins conflict
  with everything else; install separately if the offline TTS path is needed.

### Frontend (`frontend/package.json`)

- React 18.3.1 + react-dom, Vite 5.3.1, Tailwind 3.4.4, PostCSS, Autoprefixer.
- axios 1.7.2, framer-motion 11.2.12, recharts 2.12.7, lucide (icons).
- socket.io-client 4.7.5, react-webcam 7.2.0, firebase 10.12.2.

### Mobile (`mobile/`)

Flutter + Riverpod + TFLite Flutter (optional, cross-platform).

---

## What changes from v2 → v3 (stack level)

| Area | v2 | v3 |
|---|---|---|
| ML framework | TensorFlow/Keras **and** torch (DeepFace/Flan-T5) | **PyTorch only**, ONNX for serving |
| Landmarks | MoveNet 17 keypoints | MediaPipe Holistic (21/hand + pose) |
| Realtime transport | Socket.IO + `/ws/gesture` streaming | HTTP `POST /api/recognize` whole clip |
| TTS | gTTS + Coqui | Browser `SpeechSynthesis` + optional gTTS; Coqui dropped |
| Storage default | Firebase/Firestore | SQLite (Firestore optional) |
| Auth default | Firebase JWT | Guest mode (Firebase optional) |
| LLM model | hardcoded Gemini 2.0 Flash (now retired) | `GEMINI_MODEL` env var, provider interface |

> **Model-retirement note:** Gemini 2.0 Flash is retired. v3 requires the model
> name to be an env var (`GEMINI_MODEL`) so it can be swapped without code
> changes. Verify current model names at build time.
