# SignTalk AI v3 — Progress log

After every gate, the exact command and its real output are recorded here
(PROJECT_CONTEXT Section 12.9). Measured numbers only — never a fabricated
accuracy figure (Section 0.7).

---

## Gate 0 — Phase 0 environment setup (FEAT-001)

Phase 0 scope done in this run: pinned Python environment, installable
`signtalk_core` package shell, all `config/` files with Section 13 placeholder
values, `.env.example`, and the Phase-0 docs. **No ML logic yet.**

### Interpreter

Only Python 3.13.7 and 3.12.3 are installed on this machine (no 3.11, no
conda). Per the Phase-0 fallback plan the venv was created with Python 3.12
(see docs/DECISIONS.md D2).

```text
$ py -0p
 -V:3.13 *        ...\Python313\python.exe
 -V:3.12          ...\Python312\python.exe

$ py -3.12 -m venv .venv
$ .\.venv\Scripts\python.exe --version
Python 3.12.3
```

### Install (clean, editable, with dev extras)

```text
$ .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
...
Successfully installed ... fastapi-0.111.0 mediapipe-0.10.14 numpy-1.26.4
onnx-1.16.1 onnxruntime-1.18.0 opencv-python-4.10.0.84 pydantic-2.7.4
pytest-8.2.2 ruff-0.4.10 black-24.4.2 signtalk-core-0.1.0 torch-2.2.2 ...
```

Resolved versions were then frozen to `requirements.txt`:

```text
$ .\.venv\Scripts\python.exe -m pip freeze --exclude-editable > requirements.txt
# 81 pinned entries committed
```

### Import + config checks

```text
$ .\.venv\Scripts\python.exe -c "import signtalk_core, mediapipe, torch, onnx, onnxruntime, cv2, fastapi, pydantic, yaml; ..."
signtalk_core 0.1.0
mediapipe 0.10.14
torch 2.2.2+cpu
onnx 1.16.1
onnxruntime 1.18.0
opencv 4.11.0
fastapi 0.111.0
pydantic 2.7.4

$ .\.venv\Scripts\python.exe -c "from signtalk_core.config import load_config; c=load_config(); ..."
seq 32 smooth 3 t_accept 0.85 mode strict
```

### Acceptance verification (FEAT-001)

```text
$ .\.venv\Scripts\python.exe -c "import signtalk_core, yaml, json; yaml.safe_load(open('config/signtalk.yaml')); v=json.load(open('config/vocabulary.json')); assert len([s for s in v['signs'] if s.get('demo')])>=12; print('OK demo signs:', len(v['signs']))"
OK demo signs: 12

$ .\.venv\Scripts\python.exe -c "import mediapipe, torch; print(mediapipe.__version__, torch.__version__)"
0.10.14 2.2.2+cpu
```

### What was NOT run in this Phase-0 slice

- **INCLUDE dataset was NOT downloaded.** No `prepare_include.py`, no baseline
  training, so **no INCLUDE-50 accuracy number exists yet.** The Gate-0
  baseline accuracy number from PROJECT_CONTEXT Section 10 (a measured
  INCLUDE-50 official-test figure) is deferred to a later feature; it is NOT
  reported here and must not be fabricated.
- **No model training / inference.** torch resolved to the CPU build
  (`2.2.2+cpu`); GPU/CUDA selection is a later concern (`USE_GPU=auto`).
- **The Gate-1 manual acceptance** (hold Space, sign a word, see it on screen)
  belongs to Phase 1 and is not part of this feature.

Everything above is a real command and its real output; nothing is invented.

---

## Gate 1 — Pose model + recognize_clip + FastAPI server (FEAT-003)

Scope done in this run: tiny PyTorch pose model (`PoseGRU`), the
`recognize_clip()` library function returning the exact Section 6.1 dict, and a
FastAPI server (`/health`, `/api/vocab`, `/api/recognize`, and the Phase-1
enrollment endpoints) backed by SQLite (guest mode, **no Firebase**). Contract
+ smoke tests included. Random-init model weights are acceptable for Gate 1
(accuracy is irrelevant for the skeleton — Section 0.3); **no accuracy number is
claimed.**

### Server + core tests

```text
$ .\.venv\Scripts\python.exe -m pytest server/tests core/tests -q
..................                                               [100%]
18 passed, 1 warning in 2.36s
```

(5 server tests: health, vocab-count, recognize JSON contract, recognize bad
JSON -> 422, enrollment start->clip->progress; plus the 13 FEAT-002 core tests.)

### Live smoke test

Port 8000 was already held by an unrelated local process, so the smoke server
was bound to **127.0.0.1:8010** instead (my own uvicorn only; the other process
was left untouched) and stopped afterward.

```text
$ .\.venv\Scripts\python.exe -m uvicorn server.app.main:app --host 127.0.0.1 --port 8010
INFO:     Started server process [18080]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8010 (Press CTRL+C to quit)

$ Invoke-WebRequest http://127.0.0.1:8010/health
{"status":"ok"}

$ Invoke-RestMethod http://127.0.0.1:8010/api/vocab   # count=12
[{"label":"hello","category":"greetings","demo":true},
 {"label":"thank you","category":"greetings","demo":true}, ... 12 items]

$ POST http://127.0.0.1:8010/api/recognize  (synthetic 24x solid-colour frames, fps=20)
{
  "clip_id": "00d5aeb2-e65b-4656-bd63-39d28a20e6f1",
  "decision": "reject",
  "label": null,
  "candidates": [],
  "confidence": 0.0,
  "margin": 0.0,
  "agreement": {"models": "1/1", "tta": "1/1"},
  "reject_reason": "hands_not_visible",
  "latency_ms": {"extract": 3, "models": 0, "fusion": 0, "total": 3}
}
```

The synthetic no-hands clip returning `reject` / `hands_not_visible` is the
**expected PASS**: the full Section 6.1 key set and value types are present and
correct (context.json / FEAT-003 acceptance). No MediaPipe `.task` bundle is
wired up in the skeleton, so `recognize_clip` runs in no-detector mode and every
frame is treated as no-hands; the quality gate then rejects — the contract shape
is what Gate 1 verifies, not accuracy. The server was stopped after the smoke
test (port 8010 no longer listening).

### What was NOT done in this slice

- **No real inference accuracy.** Model weights are random-init; no training ran
  and no MediaPipe model bundle was downloaded, so no recognition quality is
  claimed.
- **The web client (CaptureScreen / EnrollScreen)** and the Gate-1 manual
  hold-Space acceptance belong to a later feature (web UI); not part of
  FEAT-003.

Everything above is a real command and its real output; nothing is invented.

---

## Gate 1 — Web client (CaptureScreen + EnrollScreen) (FEAT-004)

Scope done in this run: the React + Vite + Tailwind web app under `web/`
(**no Firebase, no login wall**). A hold-Space push-to-sign capture screen that
records a whole clip, uploads the UN-mirrored base64 JPEG frame batch to
`/api/recognize`, and shows the returned label (or a reject message); plus an
enrollment screen (choose signer, pick a sign from `/api/vocab`, record N reps
with a live counter, POST each clip to `/api/enroll/clip` under a session from
`/api/enroll/start`). The v2 `frontend/` folder is untouched. **No accuracy is
claimed** — the model is still random-init (Section 0.3).

### Build (real command + output)

```text
$ cd web && npm install
added 198 packages, and audited 199 packages in 39s

$ npm run build
> signtalk-ai-web@0.1.0 build
> vite build
vite v5.3.1 building for production...
✓ 87 modules transformed.
dist/index.html                   0.40 kB │ gzip:  0.27 kB
dist/assets/index-DtZ9HyNv.css    9.70 kB │ gzip:  2.70 kB
dist/assets/index-D1QbZXN4.js   189.38 kB │ gzip: 63.46 kB
✓ built in 8.56s
# EXIT=0, web/dist/ produced
```

`web/node_modules/` and `web/dist/` are both gitignored (confirmed via
`git check-ignore`) and are not committed.

### Gate-1 manual acceptance procedure (NOT yet measured)

This is the step-by-step procedure to run the Gate-1 walking-skeleton check
(Section 0.3: "hold Space, sign one of ~5 words, a word appears on screen —
accuracy is irrelevant"). **It has not been executed here**; a webcam, a human
signer, and the MediaPipe `.task` bundle are needed, so no result is recorded
below. When run, paste the real observed outcome here — never a fabricated
accuracy number.

1. Start the API server (binds `127.0.0.1:8000`, the port the Vite dev proxy
   targets):

   ```text
   .\.venv\Scripts\python.exe -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000
   ```

2. In a second terminal, start the web dev server:

   ```text
   cd web && npm run dev
   ```

3. Open the printed URL (default `http://localhost:5173`) in a browser and
   allow webcam access when prompted. The Capture screen loads with the
   one-line consent notice and a mirrored live preview.
4. Hold **Space** (or press and hold the on-screen button). The status badge
   switches to **Recording…** and a frame counter increments.
5. Perform one of the demo signs (e.g. `hello`, `thank you`, `help`, `yes`,
   `no`) and release Space. The badge shows **Analysing…**, the UN-mirrored
   frame batch is POSTed to `/api/recognize`, and the returned `label` (or the
   reject message, e.g. "Hands weren't visible — please sign again.") renders.
6. **PASS** = a word (or a reject prompt) appears on screen for a held-and-
   released sign. Accuracy of the word is irrelevant for Gate 1 and must not be
   reported as a quality figure.

   > Reality check for the current skeleton: because no MediaPipe `.task`
   > bundle is wired up yet (FEAT-003 note above), real captures currently come
   > back as `reject` / `hands_not_visible`. The reject prompt still appearing
   > on screen demonstrates the full capture → upload → response → render loop;
   > a recognised label will appear once the detector bundle and trained
   > weights are added in a later phase.

### What was NOT done in this slice

- **No live browser run recorded.** The build gate (`npm run build` -> exit 0,
  `dist/` produced) is the automated check that ran here; the manual hold-Space
  browser check above is documented but not yet executed, so no outcome and no
  accuracy number is claimed.

Everything above is a real command and its real output; nothing is invented.

---

## Convergence gate — cross-FEAT integration verification (Phase 0+1)

Scope of this run: verify that the independently built FEATs (core features,
pose model + `recognize_clip`, FastAPI server, web client) compose correctly end
to end — no seam issues across import paths, config, or the core↔server schema.
**No new feature code; this is a verification pass.** Model weights are still
random-init and no MediaPipe `.task` bundle is wired up, so no accuracy is
claimed (Section 0.3 / 0.7).

### Core tests

```text
$ .\.venv\Scripts\python.exe -m pytest core/tests -q
.............                                                            [100%]
13 passed in 0.30s
```

### Server tests

```text
$ .\.venv\Scripts\python.exe -m pytest server/tests -q
.....                                                                    [100%]
5 passed, 1 warning in 2.46s
```

(Combined `core/tests server/tests` -> **18 passed, 1 warning**.)

### Live server smoke test

The server was started in-process via `uvicorn.Server` on **127.0.0.1:8099**
(a background thread), exercised over real HTTP with the stdlib `urllib`
client, and then stopped (`server.should_exit = True`). The `requests` package
is intentionally **not** a project dependency, so stdlib HTTP was used rather
than installing anything.

```text
$ .\.venv\Scripts\python.exe _smoke.py   # throwaway helper, removed after the run
OK  /health 200 {'status': 'ok'}
OK  /api/vocab 200 with 12 items
OK  /api/recognize 200 decision=reject reject_reason=hands_not_visible keys=all-6.1-present
SMOKE PASS
```

- `GET /health` -> 200, body `{"status": "ok"}`.
- `GET /api/vocab` -> 200 with **exactly 12** items, each `{label, category, demo}`.
- `POST /api/recognize` (synthetic 24× solid-colour base64 JPEG frames, fps=20)
  -> 200 with the **complete Section 6.1 key set** present and no extra keys:
  `clip_id, decision, label, candidates, confidence, margin, agreement,
  reject_reason, latency_ms` (with `agreement = {models, tta}` and
  `latency_ms = {extract, models, fusion, total}`). The no-hands synthetic clip
  returning `reject` / `hands_not_visible` is the expected PASS — the contract
  shape is what is verified, not accuracy.

### Web build

```text
$ cd web && npm install
up to date, audited 199 packages in 3s
# EXIT=0

$ npm run build
> signtalk-ai-web@0.1.0 build
> vite build
vite v5.3.1 building for production...
✓ 87 modules transformed.
dist/index.html                   0.40 kB │ gzip:  0.27 kB
dist/assets/index-DtZ9HyNv.css    9.70 kB │ gzip:  2.70 kB
dist/assets/index-D1QbZXN4.js   189.38 kB │ gzip: 63.46 kB
✓ built in 8.10s
# EXIT=0
```

### Result

**No cross-FEAT seam failures found.** All suites and gates passed on the first
run: core↔server import paths resolve, `load_config()` + the shared app state
wire up, the `recognize_clip` return dict matches the server's `RecognizeResponse`
(Section 6.1) with no schema drift, `/api/vocab` reads `config/vocabulary.json`
and returns 12 items, and the web client builds clean against the same contract.
No source fixes were required, so no code was changed in this pass; only this
log entry was added. The throwaway `_smoke.py` helper was deleted after use.

Everything above is a real command and its real output; nothing is invented.
