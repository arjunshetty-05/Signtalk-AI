# Architecture

Source: `PROJECT_CONTEXT.md` Sections 5–7 for the **target (v3)**; the repository
itself for the **current (v2 reference)** architecture. Both are documented here
so the gap is explicit.

---

## Part 1 — Target architecture (v3, accuracy-first)

### Big idea in plain words

The browser is a **smart camera**: it checks the setup, decides when a sign
starts and ends, and uploads the whole clip. The server is the **expert**: it
extracts hand/body detail from every frame, runs several models, combines them
carefully, and only announces a word when it is sure — otherwise it asks the user
or says "please sign again". Words are then assembled into a sentence,
translated and spoken.

### Diagram

```
 BROWSER (React web app)                     SERVER (FastAPI, Python 3.10/3.11)
 ----------------------                      ---------------------------------
 Webcam 720p/1080p @30fps
   |                                          POST /api/recognize  (whole clip)
 PRE-FLIGHT CHECK (light MediaPipe Web)          |
   hands visible? light? framing?                1. decode every frame
   |                                             2. LANDMARK EXTRACTOR (MediaPipe
 CAPTURE CONTROLLER                                 Tasks Holistic, high accuracy)
   auto start/stop on hand raise/rest           3. QUALITY GATE
   OR hold Space (push-to-sign)                 4. FEATURE BUILDER (shared lib)
   |                                            5. MODELS: M1 pose-TCN/GRU
 clip buffer (frames or WebM)                      M2 pose-Transformer
   | upload ----------------------------->         M3 RGB-crop video (optional)
                                                 6. TTA (several time/scale views)
 UI shows "Analysing..."                         7. CALIBRATE + FUSE (+ context)
   |                                             8. DECISION ENGINE
 RESULT  <-------------------------------          accept | confirm | reject
   ACCEPT  -> word chip appears                  9. log clip id, probs, decision
   CONFIRM -> top-3 chips, user taps
   REJECT  -> "please sign again"
   |
 SENTENCE BOX (editable)
   words collect; pause 2.5 s or "Finish"     POST /api/compose  (sentence layer)
   |  --------------------------------------->   scripted table -> LLM -> template
 SUBTITLES + SPEECH (browser voices)
   |
 MIC -> /ws/speech -> VAD + Whisper -> live captions (Direction B)
 ENROLL MODE -> /api/enroll/* -> quality gate -> store -> fine-tune -> report
```

### The 12-lever accuracy stack

The v3 design's accuracy comes from stacking these, each measured in isolation:

1. **L1 Engineer the vocabulary for distinctness** — drop confusable look-alike
   sign pairs (read the confusion matrix).
2. **L2 Controlled capture + pre-flight check** — the app refuses to record
   until framing/light/distance are good.
3. **L3 Record-then-recognise (whole clip)** — classify a complete sign once,
   not sliding 30-frame windows (which "flap").
4. **L4 Full-quality video to the server** — heavy landmark settings and RGB
   models that need real pixels.
5. **L5 Rich features (hand detail + RGB crops)** — 21 hand landmarks per hand,
   not just wrists.
6. **L6 Start from pretrained models** — OpenHands/INCLUDE pose checkpoints.
7. **L7 Personal enrollment** — the single biggest lever for Demo-100; signers
   record their own examples and models are fine-tuned on them.
8. **L8 Ensemble of diverse models + calibrated fusion** — pose + RGB fusion is
   consistently state-of-the-art.
9. **L9 Test-time augmentation (TTA) and voting** — several views per clip.
10. **L10 Decision engine (accept / confirm / reject)** — converts high raw
    accuracy into 100% *accepted* accuracy by abstaining when unsure.
11. **L11 Context prior ("script / conversation mode")** — narrow candidates by
    conversation context; must be disclosed, off by default.
12. **L12 Regression guard (golden acceptance test)** — frozen held-out clips
    that must always pass before a commit.

### Target repository layout (v3)

```
signtalk/
  docs/            VISION.md PROGRESS.md DECISIONS.md EVALUATION.md
  config/          signtalk.yaml vocabulary.json scenarios.json
                   sentences.json phrasebook.json
  core/signtalk_core/
      landmarks.py features.py quality.py augment.py
      models/      pose_tcn.py pose_transformer.py rgb_video.py
      calibration.py fusion.py decision.py tta.py recognize.py
  core/tests/      unit + golden-vector + acceptance
  training/        prepare_include.py train.py evaluate.py export_onnx.py
                   finetune_enrolled.py collect_clips.py notebooks/
  tools/           pick_thresholds.py pick_vocabulary.py make_report.py
  server/app/      routers, schemas, providers, storage, jobs
  web/src/         React + Vite + Tailwind: preflight, capture, enroll, sentence
  data/            (gitignored) raw/ processed/ enroll/ golden/ feedback/
  models/          (gitignored/LFS) *.onnx labels.json model_card.md
  scripts/         setup, run_all, lint
```

**Key rule:** all ML logic lives in the installable `signtalk_core` package.
Training and serving import the **same** feature code — never re-implemented.

### Interface contracts (v3, change only with human approval)

| Endpoint | Purpose |
|---|---|
| `POST /api/recognize` | Whole clip (WebM/MP4 or base64 frames) → `{clip_id, decision, label, candidates[], confidence, margin, agreement, reject_reason, latency_ms}`. `decision` ∈ `accept` / `confirm` / `reject`. |
| `POST /api/confirm` | `{clip_id, chosen_label}` — user tap, stored as a labelled sample. |
| `POST /api/compose` | `{words[], emotion, history[], scenario_id}` → `{sentences{en,hi,kn}, source, verified, latency_ms}`. `source` ∈ `scripted`/`llm`/`template`. |
| `WS /ws/speech` | server → client `{type:"transcript", text, is_final, lang}`. |
| `POST /api/enroll/*` | `start`, `clip`, `progress/{id}`, `train`, `report/{id}`. |
| `GET /api/vocab` | `[{label, category, demo}]`. |
| `GET /health`, `GET /metrics` | liveness and per-stage latency histograms. |

Python library contract: `recognize_clip(frames, fps, signer_id, scenario_id=None) -> dict`
shaped like the `/api/recognize` response. One config file, `config/signtalk.yaml`,
holds every tunable.

### Decision engine (L10) — the heart of the accuracy claim

```
ACCEPT  if fused top-1 prob >= T_accept
            AND margin (top1 - top2) >= M_accept
            AND ensemble members agree on top-1 (>= k of n)
            AND TTA agreement >= A_accept
            AND clip quality gate passed
CONFIRM if not accepted but top-1 >= T_confirm: show top-3 chips to tap
REJECT  otherwise: "Not recognised — please sign again"
```

Thresholds come from the validation risk-coverage curve (`tools/pick_thresholds.py`),
then are verified once on the untouched test session. **Never tune on test.**

### Sentence layer — determinism first

1. **Scripted table** (`config/sentences.json`) — native-speaker-verified
   en/hi/kn for known word sequences. No LLM, no hallucination risk.
2. **LLM** — one call returning strict JSON `{en, hi, kn, tone}`, with a safety
   check that every signed word appears in the English sentence (else fall
   back). Model name from env `GEMINI_MODEL`; 4 s timeout; provider interface.
3. **Template fallback** — capitalise/punctuate/join + phrasebook translation.

---

## Part 2 — Current architecture (v2 reference, what the repo contains today)

This is the previous team's codebase. Per `PROJECT_CONTEXT.md` 0.2 it is
**reference only** — its "locked contracts" are void and Section 6 (above)
replaces them. It is documented here so the current code is understood, not as a
spec to preserve.

### Dataflow (as built)

```
WEBCAM frames (20-30 fps, base64 JPEG)        MIC audio (WebM/Opus)
   | WS /ws/gesture                              | WS /ws/speech
   v                                             v
MoveNet Thunder (17 keypoints)               Whisper Small STT
   v                                             v
normalise + smooth -> 30-frame rolling       live hearing subtitles
   buffer                                        v
   v                                          conversation history
BiLSTM classifier (128u, dropout 0.3)
   | top label, debounced
   | (3+ matching preds, 1.5 s cooldown)
   +-> DeepFace emotion (7 classes)
   v
NLP correction (Gemini 2.0 Flash / Flan-T5 offline)
   v
grammatical sentence
   +-> translation (Google Translate / phrasebook)
   +-> TTS (gTTS / Coqui offline)
   v
React subtitle bar
```

### Backend module map (as built)

- `backend/api/main.py` — FastAPI entrypoint; wires routers behind Firebase-JWT
  auth, CORS (two named origins), slowapi rate limiting; loads MoveNet at
  startup; `/health` is the only unauthenticated route.
- `backend/api/socket_manager.py` — wraps the FastAPI `app` in a Socket.IO
  ASGIApp (does not replace it).
- `backend/api/<domain>/router.py` — thin routers for `ai`, `analytics`,
  `emotion`, `pose`, `speech`, `translation`, `websocket`.
- `backend/api/core/` — `config.py`, `exceptions.py`, `limiter.py`,
  `metrics.py`, `offline_mode.py`.
- `backend/api/firebase/firebase_client.py` — Firebase/Firestore client
  (with a mock mode when no service account is present).
- Top-level ML modules: `classify.py`, `emotion.py`, `nlp_correction.py`,
  `speech.py`, `tts.py`, `translation.py`, `keypoint_utils.py`,
  `train_bilstm.py`, `convert_to_tflite.py`, `offline_inference.py`.
- `backend/dataset_tools/` — INCLUDE/WLASL/emotion download and preprocessing,
  feedback queue, personal-dataset recording.

### Frontend map (as built)

- `frontend/src/App.jsx`, `main.jsx`, `firebase.js`.
- `components/`: `WebcamView.jsx`, `SubtitleBar.jsx`, `ConversationHistory.jsx`,
  `AnalyticsPanel.jsx`, `LanguageSelector.jsx`, `LoginScreen.jsx`.
- `hooks/`: `useGestureSocket.js`, `useConversationSocket.js`.
- `context/`: app/auth context.
- React 18 + Vite + Tailwind + Framer Motion + Axios + Recharts + socket.io-client
  + react-webcam + firebase.

### Mobile (optional, as built)

Flutter app under `mobile/lib/` (`models/`, `providers/`, `screens/`,
`services/`, `theme/`, `widgets/`) with Riverpod and TFLite. v3 lists a native
mobile app as a non-goal.

### How the two architectures differ (summary)

| Dimension | v2 (current code) | v3 (target) |
|---|---|---|
| Landmarks | MoveNet, 17 keypoints (wrists, no fingers) | MediaPipe Holistic, 21 landmarks per hand + pose subset |
| Inference | Streaming sliding 30-frame window over WS | Record-then-recognise a whole clip over HTTP |
| Models | Single BiLSTM | Ensemble (pose-TCN/GRU + Transformer + optional RGB) |
| Confidence | Debounce (3 preds + 1.5 s cooldown) | Calibration + fusion + TTA + accept/confirm/reject |
| Adaptation | None | Personal enrollment + fine-tune |
| Output discipline | Emits a label when debounced | Abstains/asks when unsure (Demo-100) |
| ML code location | Scattered top-level modules | One installable `signtalk_core` package shared by train + serve |
