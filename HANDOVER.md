# SignTalk AI — Team Handover

A quick guide to run the app and review what's built so far. For the full vision
read `PROJECT_CONTEXT.md`; for the deeper docs see the `docs/` folder
(`ARCHITECTURE.md`, `TECH_STACK.md`, `PROJECT_PLAN.md`, `PROGRESS.md`,
`RUNBOOK.md`, `DECISIONS.md`).

---

## 1. What this is

A web app that turns **sign language → spoken + subtitled sentences** (English /
Hindi / Kannada), and also does **voice → live captions**. Accuracy-first "v3"
build: record a whole sign clip, send it to a local server, recognise it with an
ensemble, and only speak a word when confident (otherwise ask or decline).

The old `backend/`, `frontend/`, `mobile/` folders are the **previous team's v2
code — reference only, not used by v3.** The live code is in `core/`, `server/`,
`web/`, `training/`, `tools/`, `config/`.

---

## 2. What is trained so far (current state)

- **Dataset:** INCLUDE **Greetings** category only (downloaded locally).
- **9 signs trained:** `Hello`, `Thank you`, `Good Morning`, `Good afternoon`,
  `Good evening`, `Good night`, `How are you`, `Alright`, `Pleased`.
- **Models (ensemble):** a BiGRU (M1) and a small Transformer (M2), both on
  pose+hand landmark features.
- **Validation accuracy:** GRU **93.9%**, Transformer **100%** top-1.
  > ⚠️ Honest caveat: this is on a **random split** of 190 clips, which likely
  > has some data leakage (same signer in train + val). It proves the pipeline
  > learns correctly — it is **not** a defensible generalisation number yet.
  > The paper-grade number needs the official INCLUDE train/test split and/or
  > enrolled-signer testing. Live webcam accuracy will be **lower** than these
  > numbers because of the domain gap (different signer / camera / lighting).
- **Decision thresholds:** picked from the validation risk-coverage curve —
  zero accepted errors at ~94% coverage (written into `config/signtalk.yaml`).

Trained artifacts live in `runs/` and `models/` (gitignored — not in the repo;
you regenerate them by training, see section 6).

---

## 3. One-time setup

Prereqs: **Python 3.12**, **Node.js 18+**, an **NVIDIA GPU** is nice but not
required (CPU works, just slower).

```powershell
# from the repo root
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# (optional) GPU PyTorch — skip for CPU-only
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Gemini translation SDK path uses plain HTTP (requests) — nothing extra to install.

# web deps
cd web; npm install; cd ..

# landmark model bundles (pose + hand)
.\.venv\Scripts\python.exe tools/download_mediapipe_bundle.py
```

Then create `.env` from `.env.example` (it's gitignored). Minimum to run:

```
POSE_MODEL_PATH=models/pose_landmarker.task
HAND_MODEL_PATH=models/hand_landmarker.task
RECOGNIZER_MANIFEST=models/ensemble.json      # only if you have trained models
LLM_PROVIDER=gemini
GEMINI_API_KEY=<paste a key from https://aistudio.google.com/apikey>
GEMINI_MODEL=gemini-2.5-flash
```

> Without a trained model (`RECOGNIZER_MANIFEST` unset or `models/ensemble.json`
> missing) the server still runs, but recognition uses a random fallback — the
> UI works, the predicted word is just not meaningful. Train first (section 6)
> or get the `runs/` + `models/ensemble.json` from whoever trained.

---

## 4. How to run it (every time)

Use **two terminals**. Run only **ONE backend at a time** (multiple servers
starve the CPU-bound landmark step and make it hang).

```powershell
# Terminal 1 — backend
.\.venv\Scripts\python.exe -m uvicorn server.app.main:app --port 8000

# Terminal 2 — frontend
cd web
npm run dev
```

Open **http://localhost:5173**.

Healthy backend startup logs show:
`loaded ensemble: 2 members` and `Application startup complete`.

---

## 5. How to test (3 tabs)

- **Capture** (sign → voice): hold **Space**, perform one of the 9 trained signs,
  release. Wait **~2–3 seconds** (record-then-recognise is deliberately not
  instant). You get: the word (accept), top-3 "did you mean?" chips (confirm),
  or "please sign again" (reject). Accepted/confirmed words build a sentence you
  can **Speak** in en/hi/kn.
- **Captions** (voice → text): needs `faster-whisper` installed
  (`pip install faster-whisper`); otherwise it says STT is unavailable.
- **Enrollment**: record your own clips per sign (this is how we'll push demo
  accuracy up later — see `PROJECT_CONTEXT.md` L7).

Tips: good front lighting, both hands + shoulders in frame, plain background,
~1 m from camera, keep signs short (~2 s). Follow the green pre-flight lights.

---

## 6. Dataset + how to (re)train

### Dataset: INCLUDE (Indian Sign Language)

- **Download page (Zenodo):** https://zenodo.org/records/4010759
- **Official code / paper:** https://github.com/AI4Bharat/INCLUDE
- License: CC-BY-4.0. It is **56 GB total** (15 categories). **Do NOT download it
  all.** Grab one or a few category zips. We used **Greetings** (`Greetings_1of2.zip`
  + `Greetings_2of2.zip`, ~2.8 GB).
- Unzip so the layout is one folder per sign under a category, e.g.
  `data/raw/include/Greetings/48. Hello/*.MOV` (the preprocessor strips the
  `48. ` prefix and reads `.MOV`/`.mp4` automatically). `data/` is gitignored.

### Retrain / add signs (full steps in `docs/RUNBOOK.md`)

```powershell
# 1. videos -> features (slow; MediaPipe per frame)
.\.venv\Scripts\python.exe training/prepare_include.py --include-root data/raw/include --out-dir data/processed/include

# 2. train both ensemble members (uses GPU automatically)
.\.venv\Scripts\python.exe training/train.py --processed-dir data/processed/include --arch gru         --out runs/m1_gru
.\.venv\Scripts\python.exe training/train.py --processed-dir data/processed/include --arch transformer --out runs/m2_tf

# 3. pick decision thresholds from validation
.\.venv\Scripts\python.exe tools/pick_thresholds.py --processed-dir data/processed/include --manifest models/ensemble.json --write-config
```

After training, update `models/ensemble.json` and `config/vocabulary.json` so
the `labels` list matches the trained classes **in label-index order** (see the
`label_map.json` written next to each run). Then restart the backend.

---

## 7. Tests / sanity check

```powershell
.\.venv\Scripts\python.exe -m pytest core/tests server/tests -q    # should be all green
.\.venv\Scripts\python.exe -m ruff check core/ server/ training/ tools/
```

---

## 8. What's done vs what's left

**Done (code + verified):** full v3 pipeline — capture, pose+hand landmarks,
feature builder, ensemble + calibration + fusion + TTA, accept/confirm/reject
decision engine, sentence layer (scripted → Gemini → template) with en/hi/kn +
browser speech, voice→captions, enrollment recording, SQLite storage (no
Firebase), threshold tool. One real trained model (9 Greetings signs).

**Left (mostly data + manual, not code):**
- Train on more INCLUDE categories to grow the vocabulary.
- Record **enrollment** clips (our own signers) — biggest accuracy lever.
- Use the **official INCLUDE train/test split** for a defensible accuracy number.
- Native-speaker **verify Hindi/Kannada** sentences (current ones are
  placeholder machine translations — see `config/sentences.json`).
- Confirm Hindi/Kannada **voices** exist on the demo laptop.
- Freeze a golden set + acceptance test; pick final demo signs + scenario.

---

## 8b. Known behaviour: live recognition is weak (domain gap)

If every live sign comes back as the same word (we saw "How are you") or as
"please sign again", that is **expected right now** and is NOT a model bug:

- The model is **verified correct**: it predicts 18/18 right on the actual
  INCLUDE training clips (any resolution/framing). The problem is purely that a
  live webcam sign doesn't look like INCLUDE's deaf signers' gestures, so the
  model gets an input unlike anything it trained on and falls back to a
  low-confidence guess.
- Thresholds are now set to **honest defaults** (`t_accept 0.85`, `m_accept
  0.3`) so the system **abstains** ("please sign again") instead of speaking a
  low-confidence wrong word. Earlier they were auto-set to 0.5/0.0 from a leaky
  validation split, which accepted coin-flips.

**The real fix is enrollment** (PROJECT_CONTEXT L7): record our own signers'
clips for each sign and fine-tune. That closes the domain gap and is the single
biggest accuracy lever. Until then, treat live recognition as a tech demo of the
pipeline, not an accurate recogniser. To sanity-check that the model itself
works, run the training-clip test in section 7 or feed a real INCLUDE `.MOV`.

## 9. Gotchas

- **"Analysing…" forever** = usually a second/stale backend running. Kill extra
  servers; keep exactly one on port 8000.
- **GPU shows no usage during analyse** — normal. Landmark extraction is
  CPU-only (MediaPipe); the GPU is only used during training.
- **Model only knows 9 Greetings signs** right now — anything else will be
  rejected or guessed. That's expected.
- `.env`, `data/`, `runs/`, `models/*.task`, `models/*.pt`, `.venv/`,
  `web/node_modules/` are all **gitignored** — they don't travel with the repo;
  each person sets them up locally.
