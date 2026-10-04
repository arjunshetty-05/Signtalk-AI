# SignTalk AI

Real-time sign language recognition with emotion fusion, NLP sentence
correction, multilingual speech/translation, and an offline "airplane-mode"
fallback path. Originally scoped across four workstreams (ML core,
intelligence layer, backend/deployment, frontend/mobile) that share a set of
locked contracts so each piece is independently swappable — now maintained
solo; the workstream split below still describes which files belong to
which concern, useful context even without separate owners.

## Current status (as of the last working session)

- **Web (React + FastAPI backend): fully working end-to-end.** Auth,
  live webcam gesture recognition, sentence correction, Firestore-backed
  conversation history + analytics, text-to-speech, and an offline-mode
  toggle are all live and tested.
- **Gesture classifier**: trained on INCLUDE (ISL), curated down to a
  **40-word vocabulary** (`backend/runs/exp_top40_v2/`) for **64.6%
  validation accuracy** (up from an earlier 62.9% run, `backend/runs/exp_top40/`,
  via a `Dense(64)` bottleneck before the softmax head and lighter
  regularization — see "Known gotchas" below) — the full 262-word model
  (`backend/runs/exp1/`) only reaches 6.5% given ~13-16 examples/class, so
  the 40-word one is what the live server actually points at (see
  `backend/.env`). Live recognition quality depends on two timing/orientation
  details that turned out to matter a lot in practice — see "Known gotchas"
  below.
- **Translation**: code is correct (fixed a bug where it called Google's
  ADC-authenticated client library instead of using the configured API key
  via the REST endpoint), but there's no real `GOOGLE_TRANSLATE_API_KEY`
  yet — Cloud Translation API requires GCP billing to be enabled, which hit
  an unresolved Google-side billing error (`OR_BACR2_44`) last attempt.
  Translation degrades gracefully (returns original text) rather than
  crashing, but doesn't actually translate yet.
- **Mobile (Flutter)**: code-level bugs are fixed (stale-token reconnect,
  camera-handle leak, offline-mode cold-start race), and the Flutter SDK is
  installed locally (`C:\flutter`) — but the Android SDK/emulator setup
  (needs Android Studio, an interactive GUI install) was deliberately not
  done, since the web app covers the demo need. `flutter doctor` will show
  exactly what's still missing.
- **Datasets**: only INCLUDE is downloaded/preprocessed (all 15 categories,
  4,257 sequences). WLASL, FER2013, RAF-DB, NPTEL, Samanantar, and
  AI4Bharat Indic-TTS were deliberately not pursued — none has a training
  script wired up yet, so downloading them wouldn't unlock anything today.

### Known gotchas (worth remembering before touching the live pipeline again)

- **Webcam must NOT be mirrored.** `frontend/src/components/WebcamView.jsx`
  deliberately omits react-webcam's `mirrored` prop — it flips the actual
  captured screenshot pixels, not just the CSS preview, which would
  mismatch INCLUDE's unmirrored training videos and quietly hurt
  recognition. Don't re-add it for UX reasons without accounting for this.
- **`/ws/gesture` needs `frame_skip=2`, not the default 1.** INCLUDE clips
  average ~2.9s (measured directly against real clips); training resamples
  each *entire* clip to 30 frames, while live inference just slides a
  30-frame window over incoming frames. At the frontend's ~20fps capture,
  `frame_skip=1` only covers ~1.5s — `frame_skip=2` covers ~3s, much closer
  to real sign duration. This is set in `useGestureSocket.js`'s WS URL.
  Live accuracy at this setting hadn't been re-validated by a full test
  pass as of the last session — worth confirming first thing next time.
- **On-the-fly keypoint augmentation (rotation/scale/jitter/time-warp,
  `train_bilstm.py --augment`) hurt validation accuracy on this dataset,
  every magnitude tried.** Three configs (heavy: dropout 0.4/L2/label
  smoothing → 56%; light: dropout 0.3, no L2/smoothing → 51%; very light,
  no time-warp → 62%) all landed below the 64.6% no-augmentation run with
  the same architecture. With only ~666 training sequences across 40
  classes (~15-17/class) and a validation set drawn from clean,
  un-augmented clips, augmenting the training distribution pulls it away
  from what validation actually measures faster than it buys
  generalization — the usual "augmentation helps small datasets" intuition
  doesn't hold here without either far more epochs to compensate or a
  larger dataset. The flags are still there (`--augment`,
  `--aug_rotation_deg`, etc.) if per-class example counts grow enough to
  revisit this, just don't reach for `--augment` by default.

## Project structure

```
signtalk-ai/
├── backend/
│   ├── keypoint_utils.py        # MoveNet extraction, normalization, smoothing, buffering (Person A)
│   ├── classify.py              # classify_sequence() — single source of truth (Person A)
│   ├── main.py                  # standalone flat FastAPI app (Prompt A1, local ML-only dev/testing)
│   ├── train_bilstm.py          # BiLSTM training pipeline, signer-split, TFLite export (Person A)
│   ├── convert_to_tflite.py     # SavedModel -> float16 TFLite, MoveNet TFLite downloader (Person A)
│   ├── offline_inference.py     # OfflineGesturePipeline — zero-network on-device inference (Person A)
│   ├── dataset_tools/           # record_samples, dataset_stats, feedback_queue, build_retraining_dataset,
│   │                             #   download_include/wlasl/emotion/indic_nlp/indic_tts,
│   │                             #   preprocess_include (video -> keypoint sequences) (Person A)
│   ├── emotion.py               # DeepFace emotion analysis + fusion engine (Person B)
│   ├── nlp_correction.py        # Gemini 2.0 Flash -> Flan-T5-Small fallback sentence correction (Person B)
│   ├── speech.py                # Whisper Small streaming transcription (Person B)
│   ├── tts.py                   # gTTS / Coqui TTS (Person B)
│   ├── translation.py           # phrasebook -> cache -> Firestore -> Google Translate (Person B)
│   ├── phrasebook.json          # offline EN/HI/KN phrasebook
│   ├── api/                     # restructured, secured backend (Person C)
│   │   ├── main.py              # FastAPI app: routers, CORS, rate limiting, error handling
│   │   ├── socket_manager.py    # Socket.IO wrapping (deployment entrypoint)
│   │   ├── core/                # config, exceptions, metrics, limiter, offline_mode
│   │   ├── auth/                # Firebase JWT dependency
│   │   ├── pose/                # gesture pipeline connection state (reuses Person A's code)
│   │   ├── emotion/, ai/, speech/, translation/, analytics/   # REST routers
│   │   ├── websocket/           # /ws/gesture, /ws/speech
│   │   └── firebase/            # Firestore/Storage client
│   ├── firestore.rules
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── .env.example
├── frontend/                    # React web dashboard (Person D)
└── mobile/                      # Flutter mobile app (Person D)
```

## Locked contracts (do not change signatures without updating every caller)

- `classify_sequence(sequence: np.ndarray)` — input `(30, 17, 2)`, output `{"label": str, "confidence": float}`
- `/ws/gesture` emits `{"label": str, "confidence": float, "timestamp": float}`, debounced (3+ agreeing frames, 1.5s cooldown), plus `{"type": "corrected_sentence", "sentence": str, "source": "gemini"|"flan-t5", "low_confidence": bool}`
- `correct_sentence(gesture_tokens, emotion, conversation_history, force_offline=False)` → `{"sentence": str, "source": str, "low_confidence": bool}`
- `translate_text(text, target_lang, offline=False)` → `str`
- `synthesize_speech(text, lang, mode)` → raw audio `bytes`
- 17 keypoints / MoveNet **Thunder** (not 33) — locked team-wide

## Local dev setup

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in real API keys / Firebase service account path

# Run the restructured, secured backend:
uvicorn api.socket_manager:socket_app --reload

# Or, run docker-compose (production-shaped, needs secrets/firebase-service-account.json):
docker-compose up --build
```

Swagger docs: `http://localhost:8000/docs`

## Training the gesture classifier

1. Collect data: `python dataset_tools/record_samples.py --class HELLO --signer_id s01 --batch 5`
2. Check balance: `python dataset_tools/dataset_stats.py --labels_csv data/labels.csv`
3. Train: `python train_bilstm.py --data_dir data/sequences --labels_csv data/labels.csv --output_dir runs/exp1`
4. Convert to TFLite (if not already exported by step 3): `python convert_to_tflite.py --saved_model runs/exp1/saved_model --output runs/exp1/bilstm.tflite`
5. Cache MoveNet's TFLite build: `python convert_to_tflite.py --download_movenet`
6. Point the live server at the trained model:
   ```bash
   export SIGNTALK_BILSTM_SAVEDMODEL=runs/exp1/saved_model
   export SIGNTALK_LABELS_JSON=runs/exp1/labels.json
   ```
   (`classify.py` picks these up automatically — no other code changes needed.)

**If per-class examples are too sparse for decent accuracy** (INCLUDE
averages ~13-16 examples/class across 262 classes — full-vocabulary
accuracy bottoms out around 6.5%), curate a smaller, better-represented
subset instead of retraining on everything:
```python
import pandas as pd
df = pd.read_csv("data/labels.csv")
top_k = df["class"].value_counts().head(40).index.tolist()
df[df["class"].isin(top_k)].to_csv("data/labels_top40.csv", index=False)
```
then train against `data/labels_top40.csv` — fewer classes to distinguish
between means meaningfully higher accuracy from the same data (62.9% on 40
classes vs. 6.5% on 262, in practice). Trades vocabulary breadth for
reliability; worth it for a demo, not a substitute for more data long-term.

## Datasets

Raw dataset downloads don't belong in this repo (INCLUDE is several GB,
Samanantar/NPTEL are much bigger) — download straight into whatever
environment you're training in (Colab/Kaggle), not to a laptop for
re-upload. `backend/dataset_tools/download_*.py` automate the pull; each
defaults its output dir to `/content/datasets/...` on Colab or
`/kaggle/working/datasets/...` on Kaggle, falling back to
`backend/data/raw/...` (gitignored) when run locally.

| Script | Dataset | Access | Wired into existing code? |
|---|---|---|---|
| `download_include.py --categories ...` | INCLUDE (ISL) | open, 44 files on Zenodo (~50GB for all 15 categories — pick a subset with `--categories`, or `--list` to see sizes first). **Already downloaded + preprocessed** (all 15 categories, 4,257 sequences in `backend/data/sequences/` + `backend/data/labels.csv`) as of the last session. | yes, via `preprocess_include.py` — see below |
| `download_wlasl.py` | WLASL | open, per-clip scrape (some links rot) | after preprocessing — see below |
| `download_emotion.py --raf_db_archive ...` | RAF-DB | gated — license request required | no training script yet |
| `download_emotion.py --fer2013` | FER2013 | open, via kagglehub | no training script yet |
| `download_indic_nlp.py --nptel` | NPTEL ("BhasaAnuvaad") | open, via HF `datasets` | no training script yet |
| `download_indic_nlp.py --samanantar` | Samanantar | open, via HF `datasets` | no training script yet |
| `download_indic_tts.py` | AI4Bharat Indic-TTS | open, GitHub + README-linked checkpoints | no — `tts.py` uses Coqui's own pretrained `your_tts` model, not this |

Note: an earlier version of `download_include.py` pointed at a single
`INCLUDE.zip` that doesn't exist (404) — the real dataset is 44 separate
files (verified against `zenodo.org/api/records/4010759`), which is why
`--categories`/`--list` exist.

**INCLUDE → train_bilstm.py**: `train_bilstm.py` expects `(30, 17, 2)`
MoveNet keypoint sequences + `labels.csv` (the same format
`record_samples.py` writes), but INCLUDE ships raw `.MOV`/`.MP4` clips.
`dataset_tools/preprocess_include.py` converts a downloaded INCLUDE
category into that format, reusing `keypoint_utils.py`'s
`extract_keypoints()`/`normalize_keypoints()`/`TemporalSmoother` — verified
end-to-end (real internal zip structure, synthetic-video fixture, output
consumed successfully by `dataset_stats.py`).

**Caveat that matters if you train on this**: INCLUDE's folder/file
structure (`<Category>/<idx>. <Word>/<camera-file>.MOV`) does not encode
which physical signer recorded each clip — there's no signer field
anywhere in the dataset as shipped. `train_bilstm.py` splits train/val BY
SIGNER specifically to prevent identity leakage; `preprocess_include.py`'s
`signer_id` is therefore a placeholder (`--signer_id_mode per_video`, one
signer per clip, or `per_session`, an unverified heuristic grouping
near-consecutive camera file numbers) — not real signer identity. Treat
val accuracy from INCLUDE-only training as optimistic until real signer
metadata is sourced.

**WLASL → train_bilstm.py**: still needs the same kind of
video→sequence preprocessing step; `preprocess_include.py` isn't reused
as-is since WLASL's on-disk layout differs from INCLUDE's.

The emotion/translation/TTS datasets are acquisition-only for now:
`emotion.py`, `translation.py`, `nlp_correction.py`, and `tts.py` all call
pretrained/hosted models (DeepFace, Google Translate, Gemini + Flan-T5-Small,
gTTS/Coqui) with no fine-tuning script in this repo — these downloads
matter once someone adds one.

## Offline mode ("airplane-mode demo")

Set the `X-Offline-Mode: true` header or `?offline=true` query param on any
REST call (or on `/ws/gesture` / `/ws/speech` connections). This is resolved
once, centrally, in `backend/api/core/offline_mode.py::get_offline_mode`,
and:
- `/ai/predict` skips Gemini, uses Flan-T5-Small only
- `/translate` only checks the offline phrasebook (no network/Firestore)
- `/text-to-speech` forces Coqui TTS regardless of the requested mode

On-device (zero-network) inference for the gesture pipeline itself uses
`offline_inference.py`'s `OfflineGesturePipeline`, loading
`models/movenet.tflite` + `models/bilstm.tflite` + `models/labels.json`.

## Deployment

**Backend** — Railway or Render:
- Point the service at `backend/`, using the provided `Dockerfile`.
- Set env vars from `.env.example` in the platform's dashboard (never commit
  real secrets): `GEMINI_API_KEY`, `GOOGLE_TRANSLATE_API_KEY`,
  `FIREBASE_SERVICE_ACCOUNT_PATH` (mount as a secret file),
  `FIREBASE_STORAGE_BUCKET`, `CORS_ORIGIN_WEB`, `CORS_ORIGIN_PROD`.

**Frontend** — Vercel or Firebase Hosting:
- Build command: `npm run build` (Vite). Set `VITE_API_BASE_URL` and
  Firebase web config (`VITE_FIREBASE_*`) in the platform's dashboard.

**Mobile** — distribute the Flutter build via your usual channel; point
`lib/services/api_service.dart`'s base URL at the deployed backend.

## Implementation notes (useful for picking this back up later)

- `classify_sequence()` lives in `classify.py` (imported by both the flat
  `main.py` and `api/pose/service.py`) — single source of truth. It loads
  the trained SavedModel via `tf.saved_model.load(...).signatures[...]`,
  **not** `tf.keras.layers.TFSMLayer` — that's a Keras-3-only API that
  doesn't exist on the legacy Keras 2 bundled with this project's TF 2.15.
- `/ws/gesture` sits behind `api/websocket/`, with `frame_skip` and
  `offline` query params (see "Known gotchas" above for why `frame_skip=2`
  matters). It also persists corrected sentences to Firestore and
  broadcasts them over Socket.IO — this was originally only wired into the
  unused `/ai/predict` REST route, so if conversation history/analytics
  ever look stale again, check that this side effect is still present here.
- `correct_sentence()`, `translate_text()`, and `synthesize_speech()` are
  called with the offline flag already resolved centrally by
  `get_offline_mode()` — don't add per-route header-checking. Flan-T5-Small
  needs its own simple few-shot prompt (`_build_flan_t5_prompt` in
  `nlp_correction.py`) — it can't follow Gemini's richer multi-part prompt
  and will echo instruction fragments back if given it.
- Swagger docs at `/docs`. Allowed CORS origins are `http://localhost:3000`
  and `https://signtalk.vercel.app` — dev server must run on port 3000.
  Socket.IO event name for realtime sync is `"conversation:new"`, and its
  `connect` handler verifies the Firebase token itself (never trust a
  client-supplied uid). All requests need a Firebase Auth Bearer token
  except `/health`. WebSocket auth is passed as `?token=<jwt>` (browsers
  can't set custom WS headers).
- TTS: Coqui (`TTS` package, offline mode) is an optional dependency not
  installed by default (`requirements-offline-tts.txt`, kept separate due
  to dependency conflicts) — `synthesize_speech()` falls back to gTTS if
  Coqui isn't available, rather than failing offline mode entirely.
