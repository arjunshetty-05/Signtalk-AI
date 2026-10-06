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
