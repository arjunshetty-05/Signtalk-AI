# Project Plan

Source: `PROJECT_CONTEXT.md` Sections 8–12 (data, evaluation, phases/gates,
engineering rules). Full plan is ~5 weeks; a minimum ~10-day path is included.

## Gate discipline

- Do not start a phase until the previous **gate** passes.
- A gate passes only when the check has been **run** and the **real command and
  output** are written into `docs/PROGRESS.md`. Never claim an unmeasured number.
- Build a **walking skeleton** first (thinnest end-to-end path), then improve
  accuracy step by step, measuring after every change.

## Phases and gates

| Phase | Scope | Gate |
|---|---|---|
| **0 — Setup & baseline** (2–3 d) | Repo skeleton, clean envs, download INCLUDE-50 + candidate signs, load official split, run a simple baseline. | **Gate 0:** baseline number on INCLUDE-50 official test recorded in PROGRESS.md. |
| **1 — Walking skeleton** (3–4 d) | Browser records a clip (Space-hold) → `/api/recognize` → MediaPipe landmarks → one simple pose model on 5 signs → word on screen. | **Gate 1:** a team member signs 5 words and sees them (accuracy irrelevant); handedness decision written; feature golden-vector tests exist. |
| **2 — Models & ensemble** (5–6 d) | Full preprocessing with failure report; M1 + M2 (+ M3 if GPU); pretrained init attempt; calibration; fusion; TTA; vocabulary selection; ONNX export + parity. | **Gate 2:** official-split INCLUDE-50 numbers per model and ensemble; ablation table started; ONNX parity passes. |
| **3 — Decision engine & capture polish** (3–4 d) | Pre-flight check, auto start/stop, quality gate, decision engine with validation thresholds, accept/confirm/reject UI, regression-guard scaffolding. | **Gate 3:** accepted-accuracy and coverage measured on a held-out recording set, in PROGRESS.md. |
| **4 — Enrollment & Demo-100** (5–6 d) | Enrollment UI + storage + fine-tune job + report; record sessions 1–4 for all demo signers; fine-tune; thresholds; freeze golden set. | **Gate 4:** `pytest -m acceptance` passes (accepted accuracy 100%, coverage ≥ 90%) on the held-out session; n and bound stated. |
| **5 — Language layer** (3–4 d) | Sentence buffer; scripted sentence table (native-verified); LLM provider + safety check + fallbacks; language switch; browser TTS; `/ws/speech` with Whisper + VAD; captions + history. | **Gate 5:** with no LLM key, every demo sentence still appears and speaks in all three languages; with the key, p95 compose latency recorded. |
| **6 — Evaluation & paper assets** (3–4 d) | Run the full evaluation plan; produce tables/figures; fix claims. | **Gate 6:** `docs/EVALUATION.md` complete; every number traceable to a logged run. |
| **7 — Demo hardening** (2–3 d) | Venue top-up procedure, rehearsal script, backup video, one-command startup, clear errors, on-screen lighting guide. | **Gate 7:** three clean dry runs by two people, with Wi-Fi off once. |

## Minimum path if only ~10 days remain

1. Phase 1 skeleton (days 1–2).
2. Choose 8–10 distinct signs; record enrollment (3 sessions × 30 reps) for the
   demo signers (days 3–4).
3. Train M1 + M2 (+ extra seeds), calibrate, fuse, TTA (days 4–6).
4. Decision engine + confirm UI + thresholds (days 6–7).
5. Scripted sentences + browser TTS + Hindi/Kannada verification (days 7–8).
6. Golden acceptance test + dry runs (days 9–10).

Skip for the minimum path: RGB model, pretrained integration, speech
Direction B polish, emotion. Stretch (only after Gate 7): RGB branch, sign
pictures for Direction B, grow vocabulary toward 50–100, emotion tuning.

---

## Data plan (Section 8)

- **INCLUDE dataset:** IIT Madras / AI4Bharat — 263 ISL word signs, ~4,287
  videos, official train/test split plus INCLUDE-50. Download only the needed
  categories; keep official split files. (Verify licence before redistributing.)
- **Preprocessing:** video → all frames → landmarks (same extractor as the
  server) → quality gate → features `.npz`; log every skipped clip with a
  reason; per-class hand-detection rate report.
- **Vocabulary selection:** candidates → quick model → confusion analysis →
  drop confusable pairs → final 10–20 in `config/vocabulary.json`. Keep an
  UNKNOWN set (other signs, idle motion) as negatives for rejection testing.
- **Splits (prevent leakage):** official INCLUDE split for INCLUDE-only numbers;
  enrolled data sessions 1–2 train / session 3 validation / later session 4 =
  final test (touched once); hold out one whole team member for the
  signer-independent number; freeze a GOLDEN set from a held-out session.
- **Active learning:** confirm taps / corrections go to `data/feedback/pending/`
  and are human-reviewed before joining training. Never auto-train on
  unreviewed data.

## Evaluation plan (Section 9, feeds the paper)

1. Offline INCLUDE: top-1/top-5, macro-F1, per-class table, confusion matrix on
   the official split (INCLUDE-50 and the chosen vocabulary).
2. Demo-100: metrics A–D with the risk-coverage curve on the held-out enrolled
   session; state n and the rule-of-three bound.
3. Signer-independent: hold-out-signer results (Tier 2).
4. Ablations: each lever alone and cumulatively; whole-clip vs sliding-window;
   context prior on vs off.
5. Open-set: wrong-accept rate on the UNKNOWN set (target 0 accepted wrong words).
6. Live trials: 3+ signers × 2 lighting conditions × 10 reps × every demo sign;
   aim for n ≥ 300 for the headline.
7. Latency: per-stage p50/p95 on the demo machine.
8. System: kill LLM / network / Firebase and confirm the app still recognises,
   shows and speaks the scripted sentences.
9. Report limitations plainly.

---

## Risks and fallbacks (Section 11, condensed)

| Risk | Fallback |
|---|---|
| Live accuracy below offline (domain gap) | Enrollment (L7), venue top-up, pre-flight check, constrained vocabulary. |
| Hands not detected | Quality gate + clear message, dark sleeves, front light, 30 FPS, higher resolution. |
| Auto start/stop misfires | Hold-Space push-to-sign always available; show capture state. |
| Ensemble too slow on laptop | Drop M3, reduce TTA to 3 views, run ONNX; measure first. |
| LLM unavailable / model retired | Scripted sentence table first, env-var model name, 4 s timeout, template fallback. |
| Hindi/Kannada errors | Native-speaker verification; flag LLM output as unverified. |
| Whisper weak in Kannada | Test early; smaller model; browser STT fallback; best-effort scope. |
| Overfitting / inflated numbers | Session-based splits, untouched final test, golden set, honest tiers. |
| Time overrun | Follow the minimum path; phases are ordered by value. |
| Dataset licence / consent | Do not redistribute INCLUDE clips; consent notice; gitignore data. |

## Engineering rules (Section 12, condensed)

- Small single-purpose commits; `main` always runs.
- Every `core/` module has unit tests; features/fusion/decision have
  golden-vector tests; models have ONNX parity tests.
- Training and serving import the **same** feature code — never re-implement.
- No magic numbers: tunables live in `config/signtalk.yaml`.
- No secrets in git; keys only in `.env`; maintain `.env.example`.
- Fail loudly in dev; degrade gracefully on the demo path (every external call
  has a timeout, a retry cap and a fallback).
- Record seed + config + git commit hash with every run; never report a number
  you cannot reproduce. Never tune on test data.
- Run `pytest -m acceptance` before committing anything affecting recognition.
- Accessibility: high-contrast large subtitles, keyboard-operable controls, no
  flashing.
