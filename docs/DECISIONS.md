# SignTalk AI v3 — Decisions log

Non-obvious engineering choices are recorded here (PROJECT_CONTEXT Section
12.9). Each entry states the decision and the reasoning so nobody re-debates
it. Flag anything uncertain with `[VERIFY]`.

---

## D1 — Handedness: position-based, not MediaPipe Left/Right labels

**Decision.** Assign the two detected hands to "left-of-body" and
"right-of-body" by their **horizontal position relative to the body midline**
(the shoulder-midpoint x), NOT by MediaPipe's `Left`/`Right` handedness labels.

**Reasoning.** MediaPipe's Left/Right labels are defined relative to the image
and therefore **flip when the frame is mirrored** (PROJECT_CONTEXT Section 5.3,
4.4). Mirroring tricks were explicitly rejected (Section 3, "ideas rejected").
A positional rule is stable regardless of mirroring and keeps the feature
channels consistent between training and serving.

**Companion rule.** The **uploaded** video is never mirrored. Only the
on-screen preview MAY be CSS-mirrored for user comfort; the pixels sent to
`/api/recognize` are unmirrored. Missing hand => zero landmarks + a presence
flag of 0 (Section 5.3).

**Status.** Recorded now; must be re-verified against real landmarks when the
feature pipeline is implemented (Phase 1, Gate 1).

---

## D2 — Python interpreter: 3.12 (3.11 target unavailable; NOT 3.13)

**Decision.** The project targets Python 3.11, but 3.11 is **not installed on
this machine** and installing a new interpreter was out of scope for this run.
Only Python 3.13.7 and 3.12.3 are present (`py -0p`). Per the Phase-0 plan the
fallback is 3.12, so the virtualenv at `.venv` was created with
**Python 3.12.3**.

**Reasoning.** MediaPipe Tasks and PyTorch both publish 3.12 wheels, so 3.12
is a safe fallback. Python **3.13 is avoided**: MediaPipe 3.13 wheels are
unreliable `[VERIFY]`. The clean install under 3.12 succeeded and
`import mediapipe` (0.10.14) and `import torch` (2.2.2+cpu) both work (see
docs/PROGRESS.md Gate-0).

**Consequence.** `pyproject.toml` pins `requires-python = ">=3.11,<3.13"` so
3.11 remains the preferred target if it is installed later, while 3.12 is
allowed and 3.13 is excluded. The exact resolved dependency versions are frozen
in `requirements.txt`.

`[VERIFY]` MediaPipe wheel availability/stability on Python 3.13 before ever
switching to it.

---

## D3 — Locked owner decisions (storage, auth, Firebase, RGB model)

These override spec defaults and are not to be re-litigated inside the build
loop:

- **Web-only.** The app is a responsive web app (FastAPI server + React/Vite
  client). The reference `mobile/` tree is ignored.
- **SQLite storage.** Conversations, enrollment index and the decisions log use
  SQLite behind a storage interface. `STORAGE_BACKEND=sqlite`.
- **Guest auth, NO Firebase.** Authentication is guest mode (`AUTH_MODE=guest`).
  Firebase is **not used anywhere** — no service-account credential, no
  Firestore. `.env.example` carries no required Firebase value.
- **RGB model (M3) skipped for now.** The optional RGB-crop video model
  (PROJECT_CONTEXT Section 5.5, M3) is not built in this phase, but the code is
  kept **shaped** so an RGB branch can be added later (e.g. the reserved
  `rgb_frames` config key and the `signtalk_core.models` package shell).

**Reasoning.** Owner priorities are highest accuracy and a short timeline for a
college project; Firebase adds setup cost with no accuracy benefit, and SQLite
+ guest auth are sufficient for a local demo. RGB fusion is a known accuracy
lever (Section 4) kept as a backup after the core path works.
