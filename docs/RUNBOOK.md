# SignTalk AI — Training & Demo Runbook

Every command runs from the repo root
(`c:\Users\ABHIMAN KUMAR\Documents\GitHub\Signtalk-AI`) in PowerShell, using the
project venv Python: `.\.venv\Scripts\python.exe`. Do them in order; each step
says what "done" looks like. Record real numbers in `docs/PROGRESS.md`.

---

## Step 0 — (recommended) GPU PyTorch

The venv ships CPU PyTorch. Your RTX 4050 works with the CUDA 12.1 build:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # want: True
```

CPU also works (slower). Not a blocker.

---

## Step 1 — Get the INCLUDE dataset

1. Download **INCLUDE-50** (the 50-sign subset — start here) from Zenodo:
   https://zenodo.org/records/4010759
   Official code + splits: https://github.com/AI4Bharat/INCLUDE
2. Unzip so there is **one folder per sign label**, each holding that sign's
   `.mp4` videos:

   ```
   data/raw/include/
       hello/      00001.mp4 00002.mp4 ...
       thank you/  ...
       help/       ...
   ```

   The folder name IS the label. `data/` is gitignored, so nothing is committed.

> Note: check the INCLUDE licence on Zenodo before redistributing clips. Using
> them to train locally is fine.

---

## Step 2 — MediaPipe bundle (already done)

`models/holistic_landmarker.task` is already downloaded. If it ever goes missing:

```powershell
.\.venv\Scripts\python.exe tools/download_mediapipe_bundle.py
```

---

## Step 3 — Preprocess videos → features

Turns every video into a feature `.npz` using the SAME extractor the server
uses. Writes a `manifest.csv` logging every skipped/failed clip with a reason.

```powershell
.\.venv\Scripts\python.exe training/prepare_include.py `
    --include-root data/raw/include `
    --out-dir data/processed/include `
    --model models/holistic_landmarker.task
```

**Done when:** `data/processed/include/features/<label>/*.npz` exist,
`label_map.json` is written, and the console prints `ok=<N> failed/skipped=<M>`.
Open `manifest.csv` and sanity-check the per-clip `hands_pct` — lots of low
values means the videos are hard for the detector (lighting/framing).

> This is the slow step (it runs MediaPipe on every frame of every video). For
> ~1,500 INCLUDE-50 videos expect a good while on CPU. Let it run.

---

## Step 4 — Train the two ensemble members

Train M1 (BiGRU) and M2 (Transformer). Small models; minutes–tens of minutes
each on the GPU.

```powershell
.\.venv\Scripts\python.exe training/train.py `
    --processed-dir data/processed/include --arch gru         --seed 0 --out runs/m1_gru
.\.venv\Scripts\python.exe training/train.py `
    --processed-dir data/processed/include --arch transformer --seed 0 --out runs/m2_tf
```

**Done when:** each `runs/<name>/` has `best.pt`, `metrics.json`, and
`label_map.json`. Open `metrics.json` and read `best_val_top1` — **this is your
first real accuracy number (Gate 2).** Paste the command + number into
`docs/PROGRESS.md`.

> Optional: add more members with `--seed 1`, `--seed 2` for a stronger
> ensemble (diversity helps — L8).

---

## Step 5 — Make the ensemble manifest

Create `models/ensemble.json` so the server (and the threshold tool) load your
trained members. `labels` must be the list from any run's `label_map.json`
**in index order** (keys sorted by their value):

```json
{
  "labels": ["<label0>", "<label1>", "..."],
  "members": [
    { "arch": "gru",         "checkpoint": "runs/m1_gru/best.pt", "temperature": 1.0, "weight": 0.5 },
    { "arch": "transformer", "checkpoint": "runs/m2_tf/best.pt",  "temperature": 1.0, "weight": 0.5 }
  ]
}
```

(Temperatures stay 1.0 until you calibrate; weights can stay equal.)

---

## Step 6 — Pick decision thresholds (Gate 3)

Choose `t_accept` / `m_accept` from the validation risk-coverage curve — the
loosest thresholds giving zero accepted errors. Writes them into
`config/signtalk.yaml`.

```powershell
.\.venv\Scripts\python.exe tools/pick_thresholds.py `
    --processed-dir data/processed/include `
    --manifest models/ensemble.json `
    --report runs/thresholds_report.json `
    --write-config
```

**Done when:** the report prints with `zero_accepted_error: true` and a coverage
number. If it prints `false`, two signs are confusable — note it; later drop one
(L1). Paste coverage + accepted_accuracy into `docs/PROGRESS.md`.

---

## Step 7 — Serve the trained model

Point the server at the ensemble and run it. In `.env`, uncomment:

```
RECOGNIZER_MANIFEST=models/ensemble.json
```

Then:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.app.main:app --port 8000
# in web/ (separate terminal):  npm run dev
```

Open the Vite URL, go to **Capture**, hold Space and sign one of the trained
signs. Now the recognised word reflects a real model (accept / confirm chips /
reject). Add your Gemini key to `.env` (`GEMINI_API_KEY=`) for live translation.

---

## Later — enrollment & Demo-100 (biggest accuracy lever)

1. In the app's **Enrollment** tab, each signer records ~30 reps per sign across
   **3 separate sessions** (different day/light). Clips save under
   `data/enroll/<signer>/<session>/`.
2. Fine-tune on INCLUDE + enrolled data (the `finetune_enrolled.py` script will
   be written against your real recorded layout — tell the agent once you have a
   session recorded).
3. Re-run Step 6 thresholds, freeze a golden set, run `pytest -m acceptance`.

---

## Quick reference

| Thing | Path |
|---|---|
| Venv python | `.\.venv\Scripts\python.exe` |
| Dataset in | `data/raw/include/<label>/*.mp4` |
| Features out | `data/processed/include/` |
| Trained runs | `runs/<name>/best.pt` + `metrics.json` |
| Ensemble manifest | `models/ensemble.json` |
| Config (thresholds live here) | `config/signtalk.yaml` |
| Env (keys, model paths) | `.env` (gitignored) |
