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
