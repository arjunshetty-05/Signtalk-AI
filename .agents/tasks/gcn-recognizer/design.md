# Design — Signer-Independent ISL Recognizer via Skeleton Graph-Convolution (ST-GCN / DecoupledGCN)

Status: **revised** (round 4) — addresses every finding in the latest `design-review.json` / `design-review.md` (round-4 review: 4 MEDIUM, 2 NIT). Responses to the round-4 findings are in §13; round-3 responses in §12; round-2 responses in §11.
Scope: Indian Sign Language ONLY, isolated-word recognition. Maximize SIGNER-INDEPENDENT accuracy, measured honestly on the official INCLUDE / INCLUDE-50 test split.
Worktree: `c:\Users\ABHIMAN KUMAR\Documents\GitHub\Signtalk-AI\.worktrees\gcn-recognizer` (branch `feat/gcn-recognizer`)
Interpreter for all python/pytest/uvicorn: `c:\Users\ABHIMAN KUMAR\Documents\GitHub\Signtalk-AI\.venv\Scripts\python.exe`

---

## 1. Overview

The live recognizer collapses to a single label because the current model is a flattened-vector BiGRU/Transformer trained on 9 signs from one data source with a **random (leaky) split**. The ~94% validation number is signer-overfit: the model memorizes the handful of signers it saw, so a new webcam signer lands outside its learned distribution and every clip decodes to the nearest memorized point ("How are you"). The 374-dim engineered feature vector throws away the skeleton's graph structure, which is exactly the structure a signer-independent model needs to generalize across body shapes, camera framing, and signing styles.

This design replaces the flattened-vector classifier with a **skeleton graph-convolutional network over a 27-node MediaPipe-Holistic keypoint graph** — the architecture family (ST-GCN / SL-GCN / DecoupledGCN) that produces the top published INCLUDE and WLASL numbers and that AI4Bharat's OpenHands uses. We train and evaluate on the **official INCLUDE signer-independent split** so the headline number is defensible. We start on **INCLUDE-50** (50 signs, its own official split) to prove the architecture end-to-end before any 56 GB commitment, and we keep the whole serving app (capture UI, sentence layer, TTS, translation, SQLite, FastAPI routers, accept/confirm/reject decision engine) working by swapping only the model + feature representation behind the existing `RecognizerBundle` seam.

The investigation below resolves the five required decisions. External facts (OpenHands/INCLUDE assets, PyPI identity, download sizes) are **not** carried on prose: they are regenerated on demand by a committed, re-runnable probe (§2.0) whose latest run is checkpointed at `.agents/tasks/gcn-recognizer/probe_report.json`.

---

## 2. OpenHands viability spike — GO/NO-GO (evidence-based, re-runnable)

### 2.0 How the external facts are established (committed probe — read this first)

Every external fact in §2.1 / §4 (warm-start) / §5 (download sizes) is produced by **`tools/probe_openhands_assets.py`** (committed in this worktree). It HEADs each GitHub-release asset and the Zenodo sample zip (prints `Content-Length`), reads the live PyPI `openhands` metadata, fetches the OpenHands `decoupled_gcn.yaml` config and `pose_transforms.py` preset, checks each against an authoring-time snapshot baked into the script, and writes `probe_report.json` with a `drift` list and an `all_reproduced` flag. It makes only HEAD requests plus two small raw-file GETs — **no multi-GB download, no installs** — and never crashes on a dead link (records an `"error"` entry instead).

> **The numbers in §2.1/§4/§5 below are the authoring-time snapshot. They MUST be re-confirmed by running the probe as the FIRST implementation action; any fact the probe cannot reproduce (`drift` non-empty) is dropped from the design before the warm-start path or the download gate is attempted.** The repo is unmaintained (2021-era), so this is a real risk the probe exists to catch.

Latest probe run checkpointed in this round (`all_reproduced: true`, `drift: []`): env `torch 2.2.2+cu121`, `cuda_available: true`, Python 3.12; all three release assets HTTP 200 with the exact sizes in §2.1; Zenodo sample 200; PyPI `openhands` 1.16.0 `is_ai4bharat_slr: false`; config declares `decoupled-gcn` encoder + `fc` decoder + `num_nodes: 27` + `inward_edges` + shoulder-normalize; 27-node preset list byte-identical.

**Verdict: NO-GO on adopting OpenHands as a runtime dependency or on loading its Lightning checkpoint through the OpenHands package. GO on re-implementing the DecoupledGCN/ST-GCN architecture from scratch in `signtalk_core`, with an OPTIONAL warm-start from the public INCLUDE weights via a controlled tensor-extraction shim (treated as a best-effort bonus, not a dependency).**

### 2.1 What the probe establishes (authoring-time snapshot — re-confirm with §2.0)

Environment (probed): Python **3.12**, torch **2.2.2+cu121**, CUDA available = **True**.

OpenHands repo (`github.com/AI4Bharat/OpenHands`, Apache-2.0, last release 2021, repo unmaintained):

- **PyPI name is poisoned.** `pip install openhands` resolves to an **unrelated** package `openhands` (probe: version **1.16.0**, summary "OpenHands CLI - Terminal User Interface for OpenHands AI Agent"), NOT the AI4Bharat SLR library (`is_ai4bharat_slr: false`). Installing by that name pulls the wrong project. The only correct install path is `pip install git+https://github.com/AI4Bharat/OpenHands`, which drags in the dependency set below.
- **Dependency set is 2021-era and hostile to this stack** (`docs/requirements.txt`): `pytorch-lightning>=1.5.0`, `pytorchvideo>=0.1.1`, `hydra_core>=1.1.0`, `omegaconf>=2.1.0`, `torchmetrics>=0.5.1`, `timm`, `transformers>=4.9.1`, `albumentations`, plus unpinned `torch`/`torchvision`. PyTorch-Lightning 1.5 and pytorchvideo 0.1.1 predate torch 2.x and Python 3.12 by years. Installing these into the project venv risks downgrading torch (breaking the working CUDA build), is likely to fail to resolve on 3.12, and is hard to reverse. Not attempted against the venv by deliberate choice (high-risk, broad blast radius).
- **Pretrained INCLUDE checkpoints DO exist and download.** Base URL `https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/<file>`; probe HEAD results (HTTP 200):
  - `include_metadata.zip` — **47,041 B (≈47 KB)** → contains ONLY the official split CSVs (`Train_Test_Split/{train,test}_include{,50}.csv`).
  - `include_slgcn.zip` — **40,608,718 B (≈38.7 MB)** → `include/sl_gcn/config.yaml` + a Lightning `.ckpt`.
  - `include_stgcn.zip` — **32,294,362 B (≈30.8 MB)**.
- **The checkpoint is a PyTorch-Lightning `.ckpt`.** Loading with plain `torch.load` fails: `ModuleNotFoundError: No module named 'pytorch_lightning'` (the top-level pickle references `pytorch_lightning.*` objects). The underlying file **is** a standard torch zip archive, so the weight tensors are *expected* to be recoverable with a **targeted unpickler shim** that stubs the `pytorch_lightning.*` submodules — WITHOUT installing the real package. This claim (pickle structure, tensor recoverability) is the warm-start path's core premise and is the ONE §2.1 fact the HEAD-only probe cannot verify: it is an **UNVERIFIED assumption**. It is **to be verified** by `tools/import_openhands_weights.py` (§4), which is **NOT yet written** — until it actually downloads the zip, opens the `.ckpt`, and confirms the top-level pickle references only stubbable `pytorch_lightning.*` modules, the warm-start path stays **off** and the model trains from random init. If the structure differs when the script runs, it falls back to random init and never blocks.
- **The "SL-GCN" INCLUDE config is a `decoupled-gcn` encoder + `fc` decoder** — verified live from `examples/configs/include/decoupled_gcn.yaml`: `model.encoder.type: decoupled-gcn`, `model.decoder.type: fc` (`dropout_ratio: 0`), `graph_args.num_nodes: 27`, the exact `inward_edges` in §4, and `CenterAndScaleNormalize` with `reference_points_preset: shoulder_mediapipe_holistic_minimal_27`, `scale_factor: 1`. The `st_gcn` config is a plain ST-GCN. Both use the **identical 27-node graph** (§4).

### 2.2 Why NO-GO on adoption-as-dependency, GO on re-implementation

Adopting OpenHands as an importable runtime library would make the whole project depend on an unmaintained 2021 package whose dependency closure fights torch 2.2.2 / Py 3.12 — exactly the "project dependent on a checkpoint/library that may not load" failure mode the task says to avoid. The architecture itself is small, well-documented, and public (ST-GCN: arXiv 1801.07455; DecoupledGCN/SL-GCN: `openhands/models/encoder/graph/{st_gcn,decoupled_gcn,graph_utils}.py` are plain PyTorch `nn.Module`s with no exotic deps). Re-implementing them in `signtalk_core` costs a few hundred lines, removes every problematic dependency, and gives a model we can train, export to ONNX, and serve on the existing seam.

The public INCLUDE weights are still useful as an **optional warm-start**: trained on many INCLUDE signers, loading them into our re-implemented DecoupledGCN (via the shim + a key-remap) can only help signer-independent accuracy. But the project does **not depend** on that working — if the shim/remap fails, we train from scratch on INCLUDE and still ship. Both paths end in a working model. Fail-closed requirement satisfied.

### 2.3 Locked technology stack (do not revisit after approval)

- Model: **DecoupledGCN** (SL-GCN family) encoder + small FC decoder, re-implemented in `signtalk_core/models/`. **ST-GCN** re-implemented as the diverse second ensemble member (reuses the same graph + feature tensor). Chosen over BiGRU/Transformer because graph-conv over the skeleton is the published state of the art for signer-independent isolated SLR and directly encodes the body structure that generalizes across signers.
- Framework: **PyTorch 2.2.2** (already working, CUDA). **No** pytorch-lightning, hydra, omegaconf, pytorchvideo, timm, or transformers added.
- Keypoints: **MediaPipe** (already pinned 0.10.14), 27-node Holistic-minimal graph (§3).
- Serving: existing **FastAPI** + `RecognizerBundle` seam, **ONNX Runtime** for inference (keep the ONNX parity path).
- Data: **INCLUDE-50 first** (official split), optional scale to full INCLUDE-263.

---

## 3. New input: landmark + feature format (single source of truth)

### 3.1 The 27-node graph (verified live from `pose_transforms.py`)

`PoseSelect(preset="mediapipe_holistic_minimal_27")` selects these **MediaPipe Holistic** indices in this exact order (probe-verified byte-identical to the OpenHands source):

```
[0, 2, 5, 11, 12, 13, 14,                 # 7 pose nodes
 33, 37, 38, 41, 42, 45, 46, 49, 50, 53,  # 10 left-hand nodes
 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]  # 10 right-hand nodes
```
= **7 pose + 10 left-hand + 10 right-hand = 27 nodes**, each 2D (x, y). (MediaPipe Holistic numbering: pose = 0–32, left hand = 33–53, right hand = 54–74. Pose nodes = nose(0), L-eye(2), R-eye(5), L-shoulder(11), R-shoulder(12), L-elbow(13), R-elbow(14); each hand keeps a 10-point subset.)

This 27-entry list and its order are LOCKED: the adjacency `inward_edges` (§4) is defined over exactly these 27 ordered nodes (node 0 = nose = graph center; graph nodes 3,4 = shoulders used for normalization; node 7 = left-hand root, node 17 = right-hand root, matching edges `[6,17]`, `[7,8]`, `[17,18]`).

> **GRAPH NODE INDEX vs MediaPipe INDEX — do not conflate.** After `PoseSelect`, a *graph* node index g∈[0,26] maps to the g-th entry of the preset list. So graph node 3 = MediaPipe 11 (L-shoulder), graph node 4 = MediaPipe 12 (R-shoulder). The `inward_edges` and the shoulder-reference used by `CenterAndScaleNormalize` are expressed in GRAPH indices (3,4), never raw MediaPipe indices.

> Implementation: the preset list and the 26-edge `inward_edges` are copied verbatim into a frozen constant and unit-tested for byte-identity against the probe's captured source. The two MUST stay in the same node order.

> **Probe hardening (fix the whitespace-brittle check).** `tools/probe_openhands_assets.py` currently asserts the 27-node preset via `exact_list_present` = substring-match of ONE exact spacing of the list in `pose_transforms.py`. Trivial upstream reformatting (different internal spacing or line wrapping) would flip it to false and raise spurious drift even when the numbers are unchanged. Change the probe to **parse the preset list out of the fetched source into a list of ints and compare it to `EXPECTED["preset_27"]`** (already stored as a Python list in the script), keeping the raw-text marker (`"mediapipe_holistic_minimal_27"`) check only as a secondary signal. This is a semantic (value) comparison, not a layout comparison; `false → drift` still fails safe toward re-pinning.

### 3.2 Per-frame tensor the GCN consumes

GCN input shape is `(N, C, T, V)` (from `st_gcn.py`: `(N, in_channels, T, V)`):
- `C = 2` channels (x, y). Depth dropped (project decision that depth hurt accuracy; OpenHands pose configs are 2D).
- `T` = temporal length. OpenHands leaves `PoseTemporalSubsample` commented out; we fix `T = cfg.sequence_length` (currently 32) via the existing linear time-resample for parity across training/serving/ONNX. `T` stays a config value.
- `V = 27` nodes.

Normalization (verified from the live INCLUDE config, LOCKED to match any warm-start weights):
- `CenterAndScaleNormalize` with `reference_points_preset: shoulder_mediapipe_holistic_minimal_27`, `scale_factor: 1`: translate every node by the **shoulder midpoint** and divide by the **inter-shoulder distance** (graph nodes 3/4). This **mirrors the body-relative normalization in `features.py` (shoulder-midpoint origin, inter-shoulder scale)**; the hand-local (wrist/palm) re-normalization and the velocity/presence blocks of the 374-dim vector are **intentionally dropped** — the GCN learns spatial relations from the graph itself.
- Train-time augmentation (train split only, NOT serving): `ShearTransform(shear_std=0.1)`, `RotatationTransform(rotation_std=0.1)` (names verbatim from the OpenHands config). These are a major contributor to signer independence and part of the training recipe.

### 3.3 How this coexists with `features.py` and the golden test

The 374-dim `build_features` is frozen by `core/tests/test_features_golden.py` + `expected_features.npy`. We do **not** silently delete it. Plan:

1. Add a NEW builder `build_graph_features(landmark_seq, cfg) -> np.ndarray` of shape `[T, V=27, C=2]` in a new module `core/signtalk_core/graph_features.py`, importing the frozen 27-node preset constant and the `CenterAndScaleNormalize` logic. This is the **single source of truth** for the GCN, imported by BOTH the data pipeline (`prepare_include.py`, via §7) and serving (`recognize.py`, via §6) — satisfying the no-train/serve-skew rule.
2. Keep `features.py` + its golden test **unchanged and passing** during the transition (the old GRU/Transformer path keeps working until the GCN is proven). Once the GCN is the shipped recognizer and the old archs are removed from the manifest, `features.py` and its golden fixtures are retired in a deliberate, separate commit (not silently).
3. Add a NEW golden test `core/tests/test_graph_features_golden.py` with its own fixtures (`golden/input_landmarks_graph.npz`, `golden/expected_graph_features.npy`) generated by a committed `golden/make_graph_golden.py`, pinning the `[T,27,2]` output EXACTLY (atol 1e-6) and node count = 27. Regeneration plan: any intentional change regenerates these fixtures deliberately; CI catches accidental drift. Mirrors the existing `make_golden.py` pattern.
4. **Landmark struct extension (the riskiest step — explicit contract in §3.4).** `DualLandmarker.detect_frame` already computes the full 33-point pose (`full_pose = _landmarks_to_xy(pose, 33)`) but currently keeps only the 9-point `pose_subset` and discards the rest. The 27-node select needs the FULL 33-pose (to index graph pose nodes = MediaPipe 0,2,5,11,12,13,14) plus both full 21-point hands. So `FrameLandmarks` is extended with `full_pose: np.ndarray|None` (`[33,2]`) carried alongside the existing `pose_subset`; the two hands are already full `[21,2]`. This is **additive**: the existing 9-subset `features.py` path is untouched until retirement. `graph_features.py` consumes the extended struct.

### 3.4 `build_graph_features` explicit contract (Holistic-layout reconstruction)

This single function is where two independently-detected hands are mapped into the Holistic 33–53 / 54–74 slots and then the 27-index select is applied. It is specified as a contract and pinned by the §3.3 golden fixture (a reordered node silently corrupts every graph conv):

Input: a list of extended `FrameLandmarks`, each carrying `full_pose [33,2]` (zeros if no pose), `left_hand`/`right_hand` (`[21,2]` or `None`, already ordered into transport channels by wrist-x in `DualLandmarker`), and the two presence flags.

Per frame:
1. **Midline** = mean x of **full-pose indices 11 and 12** (L/R shoulder) — the full 33-pose, NOT the 9-subset's local indices. (`features.py::assign_handedness` uses the 9-subset's local indices 1/2 which ARE MediaPipe 11/12; `graph_features.py` recomputes from full-pose 11/12 so it is self-contained and does not depend on the subset layout.)
2. **Handedness assignment** reuses the project's rule exactly as `assign_handedness` implements it, which has two distinct cases: **one detected hand** → assigned to the left or right block by whether its wrist-x is left or right of the shoulder midline; **two detected hands** → sorted by wrist-x and the **leftmost wrist** fills the **left-hand Holistic block (33–53)**, the rightmost fills the **right-hand block (54–74)** — the two-hand path does **no** midline comparison (the leftmost-goes-left sort also resolves the rare same-side case deterministically). MediaPipe's own L/R label is ignored in both cases (consistent with the existing decision).
3. **Assemble a 75×2 Holistic frame**: rows 0–32 = `full_pose`; rows 33–53 = left-assigned hand (or zeros if absent); rows 54–74 = right-assigned hand (or zeros if absent). A per-node **validity** bit is tracked alongside (0 for every absent/zero-filled node, 1 for a detected node) — see step 6 for how it is used; it is NOT a model input channel (the `C=2` lock below).
4. **Apply the frozen 27-index `PoseSelect`** → `[27,2]` (and the parallel `[27]` validity vector).
5. **`CenterAndScaleNormalize`** (shoulder-midpoint origin via graph nodes 3/4; inter-shoulder scale; `scale_factor=1`), with the **explicit `_EPS` scale guard** below.
6. **Re-zero absent nodes AFTER centering** (see "Absent-node representation" below).

Then stack over frames and linear-resample to `T=cfg.sequence_length` → `[T,27,2]`. The golden fixture pins both a known 2-hand frame AND a one-absent-hand frame to their expected 27×2 layout so the mapping and the absent-node handling cannot silently drift.

**EPS scale guard (mirror `features.py` exactly).** `features.py` computes `scale = ||pose[L_shoulder] - pose[R_shoulder]||` and guards `if scale < _EPS: scale = 1.0` with `_EPS = 1e-6` (verified in source). `build_graph_features` MUST carry the **same** guard: the inter-shoulder distance is `||node3 - node4||`; if it is `< _EPS (1e-6)`, use `scale = 1.0` instead of dividing. Without this guard a **no-pose frame** (full-pose all-zero → shoulder nodes `(0,0)` → inter-shoulder distance `0`) produces a divide-by-zero and NaNs that propagate through every graph conv. This single guard is the §6 "no pose detected → zero-filled and flagged invalid" behavior made concrete at the feature layer.

**No-pose frame output (defined, not implicit).** A frame with no detected pose yields an **all-zero `[27,2]`** tensor (centering subtracts a `(0,0)` midpoint, scale falls back to `1.0`, so every node stays `(0,0)`) with **validity 0 for all 27 nodes**. This is a well-defined, finite, reproducible value — no NaN. The §3.3 golden fixture does not need a pure no-pose frame, but the EPS branch is covered by the one-absent-hand fixture plus a unit test that feeds an all-zero pose and asserts an all-zero finite output.

**Absent-node representation under the `C=2` lock (DECISION).** Adding per-node validity as a third input channel (`C=3`) would let the GCN distinguish "absent" from "present at the origin" cleanly, but it **contradicts the `C=2` lock** (§3.2) and would make our tensor incompatible with the 2-channel OpenHands warm-start weights (§4). The alternative — leaving a zero-filled absent node at its raw `(0,0)` and letting it flow through centering — is wrong: after subtracting the shoulder midpoint and dividing by scale, `(0,0)` moves to `(-mid/scale)`, a nonzero constant the GCN would read as a *real* hand position (a "phantom hand"). **Decision: keep `C=2`; after `CenterAndScaleNormalize`, re-zero every node whose validity bit is 0 (step 6), so an absent node sits at the normalized origin `(0,0)` — a single defined location, not a phantom position — and rely on the quality gate to reject clips with too many invalid frames.** This keeps absent nodes at one consistent, learnable location, preserves warm-start compatibility, and makes "absence" a stable signal (origin + the clip-level quality gate) rather than a per-frame channel the model would have to learn to mask. The quality gate (`hands_not_visible`, `too_short`, motion floor, §6) remains the enforcement point for "too much of this clip is invalid."

**Invariant ownership:** the EPS guard and the post-centering re-zero are owned by `build_graph_features` (the single train==serve source of truth, §3.3); the clip-level "too many invalid frames → reject" invariant is owned by the serving quality gate (§6). Validity is computed in `graph_features.py` and consumed by the quality gate; it is deliberately NOT a model input.

As part of this work, the **contradictory module docstring in `landmarks.py`** (its top docstring claims `HolisticLandmarker` works on 0.10.14 and separate landmarkers "are not needed", while the Protocol/`DualLandmarker` docstrings say the combined task crashes and separate tasks are used — the running code is `DualLandmarker`) is **corrected** so the top docstring matches reality.

---

## 4. Model architecture (concrete)

Graph (LOCKED, probe-verified verbatim from the live INCLUDE config, 27 nodes, `center=0`, GRAPH indices):

```
inward_edges = [
 [2,0],[1,0],[0,3],[0,4],[3,5],[4,6],[5,7],[6,17],[7,8],[7,9],[9,10],
 [7,11],[11,12],[7,13],[13,14],[7,15],[15,16],[17,18],[17,19],[19,20],
 [17,21],[21,22],[17,23],[23,24],[17,25],[25,26]]
```
Adjacency is built with the ST-GCN spatial partition strategy (`GraphWithPartition`, `strategy="spatial"`, `max_hop=1`) — re-implemented from `graph_utils.py` (pure NumPy, no deps).

Primary encoder — **DecoupledGCN** (`signtalk_core/models/decoupled_gcn.py`): re-implement from `openhands/models/encoder/graph/decoupled_gcn.py` (decoupled spatial graph conv with per-group adjacency + temporal conv + residual, DropGraph regularization). Decoder = FC head to `num_classes` (`decoder.type: fc`, `dropout_ratio: 0`). Input `(N,2,T,27)` → logits `(N, num_classes)`.

Second ensemble member — **ST-GCN** (`signtalk_core/models/st_gcn.py`): re-implement from `st_gcn.py` with `edge_importance_weighting: true`, `dropout_ratio: 0.05`. Architecturally diverse while sharing the exact `(N,2,T,27)` input, so the existing probability-level fusion (`fuse_probabilities`) and TTA machinery apply unchanged.

Both are small (well under 6 GB VRAM on the RTX 4050) and export cleanly to ONNX (plain Conv2d/BN/Linear ops; opset 17).

**ONNX export is a NEW branch in `export_onnx.py`, not "the same path".** Current `export_onnx.py` restricts `--arch` to `{gru,transformer}`, instantiates `PoseGRU`/`PoseTransformer`, and builds `dummy = torch.randn(1, seq_len, feature_dim())` i.e. `[1,32,374]`, input name `features`, dynamic axes `{0,1}`. The GCN needs: add `decoupled_gcn`/`st_gcn` to `--arch`; branch `_load` to the GCN modules (and NOT call `feature_dim()`); build the dummy already in GCN layout `dummy = torch.randn(1, 2, args.seq_len, 27)` (equivalently `permute_for_graph(torch.randn(1, args.seq_len, 27, 2))` using the shared §7.2 helper, so the exported axis order matches training/serving exactly); set `dynamic_axes={"features": {0: "batch", 2: "time"}, "logits": {0: "batch"}}`; keep the existing PyTorch↔ONNX parity assertion (exit non-zero on mismatch).

**Optional warm-start (bonus only — NOT yet written; an explicit deliverable of this task, not a file that already exists):** a one-off `tools/import_openhands_weights.py` downloads `include_slgcn.zip`, extracts the `.ckpt` tensors with a `pytorch_lightning.*`-stubbing unpickler shim (recover `state_dict` without installing Lightning), remaps key names to our DecoupledGCN, and loads with `strict=False`. **This script is the first and only thing that actually opens the `.ckpt`; the pickle-structure/tensor-recoverability premise in §2.1 is an UNVERIFIED assumption until this script runs.** It **first verifies** the `.ckpt` is a torch zip whose top-level pickle references only stubbable `pytorch_lightning.*` modules; if the structure differs, or any shape/key/class-count (263) mismatches, it logs and **falls back to random init** — never blocks. Applies to full INCLUDE-263 only (public weights are 263-class); for INCLUDE-50 we train from scratch (or fine-tune a 263→50 head). Gated behind a flag; off the critical path.

**Warm-start load acceptance — minimum-match guard (REQUIRED; `strict=False` alone is not an acceptance criterion).** `model.load_state_dict(remapped, strict=False)` does NOT fail on an *incomplete* remap: if the remap table covers only a subset of the module names (very likely for a from-scratch re-implementation whose names differ from OpenHands'), the call returns silently with a long `missing_keys` list and leaves most tensors at their random init, yet the warm-start would otherwise be reported as "succeeded." That is strictly worse than honest random init because it corrupts the accuracy comparison (a mostly-random model is credited to warm-start). The script therefore applies an explicit acceptance gate AFTER the load:

- Capture the returned `(missing_keys, unexpected_keys)`.
- Compute `warmstart_loaded_fraction = (# model param tensors matched and loaded) / (total # model param tensors)`, counting by **parameter tensor** (entries of `model.state_dict()` that are `nn.Parameter`/buffers the model actually uses), i.e. `1 - len(missing_keys)/len(model_state_dict)`.
- **Accept the warm-start only if BOTH hold:** (a) `warmstart_loaded_fraction >= 0.90` (at least 90% of parameter tensors by count were loaded), AND (b) **every** graph-convolution weight and temporal-convolution weight tensor in the DecoupledGCN was matched by name AND equals the expected shape (the spatial/temporal conv weights are the layers that actually carry the signer-independent signal; a warm-start that misses them is worthless even if it clears 90% on bias/BN tensors). The set of "graph-conv + temporal-conv weight" tensor names is enumerated explicitly from the DecoupledGCN module tree, not inferred by substring.
- If either condition fails: log the unmatched/misshaped keys at WARNING, discard the partial load, **re-initialize the model from random init**, and set `warmstart_loaded_fraction = 0.0`.
- **Record `warmstart_loaded_fraction` (and the pass/fail of the conv-shape check) in the run metadata** (`label_map.json`'s run block / the training run record) so every reported accuracy number states honestly whether warm-start actually happened. The fraction is also echoed to stdout at load time.

This gate is unit-testable without the real `.ckpt`: feed `load_state_dict` a deliberately-partial remap and assert the fallback-to-random path triggers and `warmstart_loaded_fraction == 0.0`; feed a complete remap and assert acceptance with fraction `>= 0.90` and the conv-shape check passing.

---

## 5. Data scope, download mechanism, and budget (USER CHECKPOINT)

**Recommendation: start with INCLUDE-50.** It has its own official signer-independent split (`train_include50.csv` / `test_include50.csv`, obtainable from the 47 KB `include_metadata.zip`), gives a defensible signer-independent number fast, and proves the GCN end-to-end before any multi-GB commitment. Scale to full INCLUDE-263 only after INCLUDE-50 clears the accuracy gate.

Download facts (probe-verified authoring-time snapshot; re-confirm via §2.0):
- Official split CSVs: **sourced specifically from GitHub `include_metadata.zip`** (**47,041 B**) at `https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_metadata.zip`, listing clips as `Category,Word,Video,FilePath` (e.g. `Animals/4. Bird/MVI_2987.MOV`). Authoritative signer-independent partition for INCLUDE and INCLUDE-50. **Source is LOCKED to this GitHub zip, NOT the Zenodo `Train_Test_Split.zip`.** The repo's `data/raw/include/download_data.sh` ends with `wget https://zenodo.org/record/4010759/files/Train_Test_Split.zip` (verified in the file), so a second split source exists on disk; the two may differ in column layout, escaping, or `FilePath` formatting, and the §7.1 join-key normalization is written against the **GitHub** CSV's `FilePath` format. Mixing sources would silently under-match. The pipeline therefore (a) consumes only the GitHub `include_metadata.zip` CSVs and (b) **asserts the CSV header is exactly `Category,Word,Video,FilePath`** (byte-exact, in that order) before trusting any join, failing loudly on mismatch (see §7.1).
- Raw videos: Zenodo record `4010759`, served as **whole-category zips** (confirmed from `data/raw/include/download_data.sh`, which lists every URL as `https://zenodo.org/record/4010759/files/<Category>_<i>of<n>.zip`). Verified size signal: `Greetings_1of2.zip` alone = **1,573,733,886 B (≈1.47 GB)**. Full INCLUDE ≈ 56 GB.

**Because Zenodo serves whole-category zips (not per-clip), an INCLUDE-50 subset still pulls every category that contains any of the 50 words — potentially most of the ~15 categories.** The "subset" is therefore NOT guaranteed to be small; with `Greetings_1of2` alone at ≈1.47 GB it can approach a large fraction of the full 56 GB if the 50 words span most categories. The earlier "single-digit GB" estimate is withdrawn.

**>>> USER CONFIRMATION REQUIRED BEFORE ANY DOWNLOAD <<<**
The pipeline will NOT download raw video until the user approves, and the gate presents a **concrete measured number, not an estimate**:

1. `tools/select_include50_downloads.py` runs FIRST (needs only the 47 KB metadata zip, no video). It parses `train_include50.csv` + `test_include50.csv`, maps each referenced `Category` to the exact set of Zenodo category zips, HEADs each zip for `Content-Length`, and prints the **category list + summed byte total** (and, since zips are whole-category, how much of that is "wasted" relative to the clips actually needed).
2. ONLY THEN is the user asked to approve one of:
   - **(A)** Download the measured INCLUDE-50 category-zip set from Zenodo, extract, keep only clips in the two CSVs. Size = the number from step 1. **Recommended.**
   - **(B)** Full INCLUDE-263 (≈56 GB, long download). Defer until INCLUDE-50 passes the gate.

Time/disk budget is dominated by (a) the download (the measured number) and (b) one-time MediaPipe pose extraction over every clip (CPU/GPU-bound, minutes-to-hours). Both are flagged at the gate; the pipeline must stop and get a yes before fetching gigabytes.

---

## 6. Serving integration

The seam is `RecognizerBundle` (built in `server/app/recognizer.py::load_bundle()` from `RECOGNIZER_MANIFEST`) consumed by `recognize_clip` in `core/signtalk_core/recognize.py`. The accept/confirm/reject decision engine, fusion, TTA, calibration, and the Section 6.1 response dict stay **unchanged in shape**. Concrete edits:

**1. New arch names in `load_bundle()`.** Extend the `arch` dispatch in `server/app/recognizer.py` (currently `gru`/`transformer`, else `ValueError`) with `"decoupled_gcn"` → `build_decoupled_gcn(...)` and `"st_gcn"` → `build_st_gcn(...)`. Manifest schema is otherwise identical (`labels`, `members[*].{arch,checkpoint,temperature,weight}`), plus one new top-level key `"feature_kind": "graph"` (default `"vector"` when absent, so old manifests keep working). Old arch names stay valid until retirement.

**2. Feature plumbing + forward adapter (the forward wrapper has no graph seam today — this spells out the edits).** Current path: `recognize_clip` → `build_tta_feature_views(seq,cfg)` → list of `[T,F]` → `_model_probs_over_views(model, views, temp, device)` which does `torch.from_numpy(np.stack(views))` (`[V,T,F]`) → `model(batch)`. `RecognizerBundle` has NO `feature_kind` field and `_model_probs_over_views` has no model-type awareness. Edits:
   - Add `feature_kind: str = "vector"` to the `RecognizerBundle` dataclass; `load_bundle` sets it from the manifest. Update the `RecognizerBundle` docstring contract (currently hard-codes "accept float32 `[B,T,F]`") to document the graph members accept `[B,2,T,27]`.
   - Add `build_tta_graph_views(seq, cfg)` in `tta.py` that reuses the SAME frame-trim logic as `build_tta_feature_views` but calls `build_graph_features` per view (returning `[T,27,2]` views). (`build_tta_feature_views` calls `build_features` internally per view, so a graph TTA path needs its own function, not a feature swap.)
   - Add `_graph_model_probs_over_views(model, views, temp, device)` that stacks views to `[V,T,27,2]`, applies the **shared `permute_for_graph`** helper (§7.2) to get `[V,2,T,27]`, then `model(batch)`; everything after (temperature, per-view top-1) is identical to the vector path. This uses the SAME permute definition as train/export/pick_thresholds (one source, no per-path reimplementation).
   - In `recognize_clip`, branch on `bundle.feature_kind`: `"graph"` → `build_tta_graph_views` + `_graph_model_probs_over_views`; else the existing vector path. The Phase-1 fallback (no bundle) stays on the vector path. Downstream fusion/decision are already shape-agnostic (operate on probability vectors).

**3. Serving `config/vocabulary.json` must be regenerated (it is a THIRD copy of label order).** Today it is 9 Greetings signs with a note "Order MUST match ... label_map.json". `recognize_clip` uses `bundle.labels` when a bundle exists but falls back to `_vocab_labels(root)` (reads `vocabulary.json`) in skeleton mode, and the sentence/UI layers read it too. An INCLUDE-50 model has 50 labels. Add a committed `tools/sync_vocabulary.py` that regenerates `config/vocabulary.json` from the trained `label_map.json` in label-index order, run as part of shipping the model.

**Exact output schema (LOCKED — must match the existing reader, not a new shape).** `recognize.py::_vocab_labels` reads `data["signs"]` then `s["label"]` per entry (verified in source: `signs = data["signs"] if isinstance(data, dict) else data; return [s["label"] for s in signs]`), and the UI/sentence layers read `category` and `demo` on each entry (the current 9 entries all carry `{label, category, demo}`). A flat list, a `{"labels":[...]}` object, or `signs` entries missing `label` would raise `KeyError`/`TypeError` at skeleton-mode startup and break any `category`/`demo` consumer. Therefore `sync_vocabulary.py` MUST emit exactly the existing object schema:

```json
{
  "note": "Auto-generated from label_map.json by tools/sync_vocabulary.py — do not hand-edit. Order matches class-index order.",
  "signs": [
    { "label": "<class name>", "category": "<Category>", "demo": false }
  ]
}
```

- `signs` is ordered strictly by `label_map.json` class index (entry *i* is the class whose index is *i*).
- `label` = the class name exactly as it appears as the `label_map.json` key for that index (byte-identical; no re-casing or prefix-stripping — the name is already post-`clean_label`).
- `category` = the INCLUDE `Category` for that word taken from the split CSV join (§7.1) when available; when the category cannot be resolved (e.g. the INCLUDE-50 run where the CSV category is not carried through), fall back to the literal string `"include50"`. Never null/missing.
- `demo` = `false` for all generated entries (these are real trained classes, not the hand-authored demo set).

Add a **three-way order test** asserting `[s["label"] for s in vocabulary.json["signs"]] == manifest.labels == list(label_map keys in index order)` — the three copies of label order must be element-for-element identical, not merely the same set.

**4. Thresholds re-tune (and `pick_thresholds.py` needs the same graph plumbing).** The current `0.85/0.3` conservative defaults were set **without a clean split** and are not calibrated to the new model. (`config/signtalk.yaml`'s own comment records that the leaky *auto-picked* values were `0.5/0.0`; `0.85/0.3` are the honest hand-set defaults.) Re-tune with `tools/pick_thresholds.py` on a VALIDATION slice carved **from the INCLUDE-50 TRAIN split only** (never the test split). But `pick_thresholds.py` as written: dispatches only `gru`/`transformer` in `_build_members` (raises on other archs), and `_fused_val_probs` does `model(x.unsqueeze(0))` on `[1,T,F]` vector features via `FeatureDataset` → so it needs the SAME graph branch as items 1–2: add `decoupled_gcn`/`st_gcn` to `_build_members`, and feed graph `.npz` with the `[1,2,T,27]` permute. It also currently derives VAL via `stratified_split` on all gathered items — on the official-split path it must instead read the VAL slice carved from the TRAIN tree only (§7), never from the test tree. `pick_thresholds.py` is therefore on the list of files the `feature_kind` change touches.

**5. One backend, port 8000, no `--reload`** (unchanged). Skeleton mode still works when no manifest/checkpoints are present (falls back to vector path).

### Error handling (concrete, per failure)

- **Missing/undecodable frame** (`decode_base64_jpeg` raises `ValueError`): fatal for that request → HTTP 4xx by the router (unchanged); logged at WARNING with clip id, no secret content.
- **No pose detected in a frame**: recoverable → that frame's 27 nodes zero-filled and flagged invalid; the existing quality gate (`hands_not_visible`, `too_short`, motion floor) rejects the clip with a reason if too many frames are empty. Returns a 6.1 `reject` dict; not logged as error (expected).
- **Checkpoint missing/corrupt at startup** (`load_bundle`): recoverable → log WARNING, fall back to skeleton/vector so the server still boots (existing behavior). No crash.
- **GCN forward shape mismatch** (feature/model disagreement): fatal programmer error → raise `ValueError` with expected-vs-actual `(C,T,V)` at the forward boundary (mirrors current dim-mismatch guards). Caught by tests before serving.
- **ONNX/PyTorch parity failure on export**: fatal at export time → `export_onnx.py` exits non-zero (existing behavior), blocks shipping a divergent model.
- **Warm-start weight load mismatch** (optional path): recoverable → log and fall back to random init; never blocks training.

### Input validation (external inputs)

- `frames`: required, non-empty list of base64 JPEG strings; empty → `ValueError` (existing).
- `fps`: required, must be > 0; else `ValueError` (existing).
- Each decoded frame → MediaPipe; nodes are normalized `[0,1]` coords; absent nodes = 0 with a validity flag.
- Manifest: `labels` required non-empty; `members` required non-empty; unknown `arch` → `ValueError` (existing, extended for new names); `feature_kind` optional, defaults `"vector"`, must be `"vector"`|`"graph"`.

### Invariant ownership

- **Feature definition parity (train == serve)**: owned by `graph_features.py` as the single importer (data pipeline + serving); enforced by the new golden test. Train/serve skew was a root cause of the collapse.
- **Node order ↔ adjacency consistency**: owned by `graph_features.py` + the graph constant; enforced by a unit test asserting the 27-preset list and the 26-edge list are byte-identical to the probe-captured OpenHands source and length-consistent. A reordered node silently corrupts every graph conv.
- **Label order ↔ checkpoint ↔ vocabulary.json**: owned by the per-run `label_map.json` (written by `train.py`); the manifest `labels` AND `config/vocabulary.json` are both regenerated from it (item 3); enforced by the three-way order test.

---

## 7. Data pipeline rewrite + honest signer-independent evaluation protocol

This is the mechanism that makes the headline number signer-independent; it is a real rewrite of `prepare_include.py`, not a one-line change. Today `prepare_include.py` has **no split concept**: `discover_clips` walks the tree, `clean_label` strips the `"NN. "` prefix, and splitting happens later and **randomly** in `dataset.py::stratified_split`. The `group_split` promised in `dataset.py`'s docstring **does not exist** — do NOT reach for it; separation comes exclusively from the official CSV split trees.

### 7.1 `prepare_include.py` — official-split routing (concrete)

Add `--split-csv-train` / `--split-csv-test` (paths to `train_include50.csv` / `test_include50.csv`, **extracted from the GitHub `include_metadata.zip`** per §5 — NOT the Zenodo `Train_Test_Split.zip`). When supplied:
- Parse both CSVs (`Category,Word,Video,FilePath`, e.g. `Animals/4. Bird/MVI_2987.MOV`). **Before any join, assert the header row is byte-exactly `Category,Word,Video,FilePath` in that order; on any mismatch, abort with a loud error naming the file** (this is the guard against someone feeding the Zenodo-sourced CSV, whose layout the join-key normalization is not written for). `select_include50_downloads.py` applies the same header assertion when it reads the CSVs to compute the download set.
- **Join key normalization** (the join is non-trivial — CSV keeps `"4. Bird"` and `.MOV` casing; `discover_clips` strips `"NN. "`): build the key from the CSV `FilePath` by taking `(Category, Word-folder, Video-stem)`, applying `clean_label` to strip `"NN. "`, lower-casing, and dropping the extension; build the same key from each on-disk clip. Match **case-insensitive and extension-insensitive**.
- **Emit TWO feature trees**: `features/train/<label>/<stem>.npz` and `features/test/<label>/<stem>.npz`, built with **`build_graph_features`** (the new `[T,27,2]` builder), plus a `split` column in `manifest.csv`.
- **Surface mismatches as errors — add a NEW count-reconciliation guard** (the current script only logs per-clip status rows to `manifest.csv`; the "4,257 vs ~4,292" string is a historical anecdote in the module docstring, NOT an existing count check): every CSV row with no on-disk match AND every on-disk clip in neither CSV is logged, and a non-zero count fails the run.
- `label_map.json` is built from the union of labels across both trees (sorted), as today.
- Flags are `--pose-model` / `--hand-model` (NOT the stale `--model holistic_landmarker.task` shown in the script's own docstring example — **fix that stale docstring example while editing the file**).

### 7.2 `train.py` + `dataset.py` (concrete edits — train.py has the SAME blockers as serving/export/pick_thresholds)

`training/train.py` is the file where the model is actually fit, and as written it cannot train a GCN member — it has the identical two blockers already spelled out for serving (§6 item 2), ONNX export (§4), and `pick_thresholds.py` (§6 item 4), and they were previously left unstated for `train.py`. Both are fixed here with the SAME transform definition so no fifth divergent permute is invented:

- **`build_arch` + `--arch` choices.** `build_arch(arch, num_classes)` currently dispatches only `"gru"`/`"transformer"` and raises `ValueError` otherwise; `--arch` has `choices=["gru","transformer"]`. Extend both: `"decoupled_gcn" → build_decoupled_gcn(num_classes)` and `"st_gcn" → build_st_gcn(num_classes)` (the same factory functions `load_bundle`/`export_onnx` call), and add both names to the `--arch` `choices` list.

- **One shared permute — `permute_for_graph`.** The train loop (`loss = criterion(model(x), y)`) and `evaluate()` (`logits = model(x)`) both feed batches straight from `FeatureDataset`, which returns `data["features"]` unchanged. For the graph pipeline that array is `[T,27,2]`, so a DataLoader batch is `[B,T,27,2]`, while the GCN forward expects `[B,2,T,27]`. There is no permute anywhere in `train.py` today. Define **one** helper — `permute_for_graph(x) -> x.permute(0, 3, 1, 2).contiguous()` (`[*,T,27,2] → [*,2,T,27]`) — in a shared module imported by train/serve/export/pick_thresholds so the axis transform has a **single definition** and cannot drift between paths. Apply it right after the batch is moved to the device, gated on arch (`arch in {"decoupled_gcn","st_gcn"}`, equivalently `feature_kind == "graph"`): in the train loop (`x = x.to(device); if graph: x = permute_for_graph(x)`) and identically inside `evaluate()` before `model(x)`. The vector path is unchanged. (`_graph_model_probs_over_views` in §6, the `export_onnx` dummy, and `pick_thresholds`'s `[1,2,T,27]` permute all call this same helper.)

- **VAL carved from TRAIN only — gated by an EXPLICIT, non-defaulting flag (no layout auto-detection).** The trigger for the official-split path must NOT be inferred from directory layout. Auto-detection is unsafe here: `gather_items(processed_dir, label_map)` globs `processed_dir/features/<label>/*.npz` (verified in source), so against the two-tree layout (`features/train/<label>/`, `features/test/<label>/`) a `processed_dir` pointed at the parent finds **zero** items (silent empty run), and a `processed_dir` pointed loosely at `.../features` could glob BOTH trees into one list and hand them to `stratified_split` — the exact random re-mixing / leakage the whole protocol exists to prevent. The single most leakage-sensitive decision cannot ride on a heuristic.

  Concrete contract: add a required-when-official **`--official-split`** flag (equivalently a `--train-tree <path>` / `--test-tree <path>` pair; the flag has **no default** and the official path is OFF unless it is passed). When `--official-split` is set:
  - `gather_items` is pointed at **`features/train` only**; it MUST NOT glob `features/test` or the parent.
  - the VALIDATION slice is carved **out of TRAIN only** via per-class `stratified_split` applied to the TRAIN items; `stratified_split` is **never** called over a combined train+test set.
  - `features/test` is **untouched** by `train.py` — it is read exactly once by the separate evaluation step (§7.3).
  
  When `--official-split` is NOT set, `train.py` keeps its current single-tree random-split behavior unchanged (used only for throwaway local experiments, never for the headline number).

  **Leakage-guard test is tied to the explicit flag, not a layout heuristic:** a test asserts that with `--official-split` set, `gather_items` is invoked against the TRAIN tree path only and `stratified_split` receives TRAIN-tree items exclusively (assert on the call arguments), and that `features/test` paths never appear in the fitting/VAL item set. This is the pipeline-routing leakage guard of §7.3; it keys off the flag being set, so a future layout change cannot silently re-enable random mixing.

- **`dataset.py`** loads `[T,27,2]` arrays unchanged (it is already feature-shape-agnostic: `FeatureDataset.__getitem__` returns `data["features"]`). Correct its docstring to note `group_split` is unimplemented and the official CSV split supersedes it.

### 7.3 Evaluation protocol

- **Headline number comes ONLY from `features/test` (= `test_include50.csv`)** — signers/clips never seen in training. NOT a random split, NOT the user's enrollment clips. The test tree is loaded **exactly once**, at the end.
- Report **top-1 and top-5** on the test split (the training loop already computes both). Also report **accepted-accuracy and coverage** at the chosen thresholds (risk-coverage), since the product abstains rather than guesses.
- **Leakage guards**: membership decided by the official CSV via the two trees; a test asserts the train/test feature trees are **path-disjoint** (necessary) AND that `stratified_split` is not on the official-split code path (sufficient to prove random re-mixing cannot sneak back in). The CSV-content disjointness check alone is NOT sufficient — the pipeline-routing test is the real guard.
- Sanity target: published INCLUDE-50 GCN results are high (SL-GCN / DecoupledGCN reports on INCLUDE-50 sit in the ~90%+ top-1 range); a correct signer-independent reimplementation should land comparably. A collapse to one label (the current live failure) must be gone.
- **Baseline being replaced:** the current model has only a leaky random-split ~94% VAL number and NO honest signer-independent figure — on the official test split its effective behavior is the observed live collapse (every webcam clip → one label, i.e. ≈ chance / one-class). The gate below is stated as absolute floors, not "beats baseline," precisely because there is no trustworthy signer-independent baseline number to beat.
- **Gate to proceed to INCLUDE-263 (concrete, checkable floors — ALL must hold):**
  1. **Official-test top-1 ≥ 70%** on `features/test` (= `test_include50.csv`), the signer-independent split. (70% is a deliberately conservative floor well below the published ~90%+ range: clearing it proves the architecture generalizes across unseen signers rather than memorizing them; falling short means the re-implementation or data pipeline is wrong and INCLUDE-263 would waste the 56 GB.)
  2. **Accepted-accuracy ≥ 85%** at the tuned thresholds (§6 item 4) on the official test split — i.e. among clips the decision engine *accepts* (does not abstain on), at least 85% are correct. This protects the product promise that an accepted label is trustworthy; coverage (accept rate) is reported alongside but is not itself gated.
  3. **Manual webcam check shows ≥ 2 distinct accepted labels** across a short live session (sign several different INCLUDE-50 words). This directly refutes the single-label collapse that triggered this whole re-architecture; one accepted label for everything = FAIL regardless of (1)/(2).
  
  All three numbers are recorded in the run metadata next to `warmstart_loaded_fraction` (§4) so the go/no-go decision is auditable. If any floor is missed, do NOT download INCLUDE-263; diagnose the pipeline first.

---

## 8. Testability

- **Unit (no MediaPipe, no torch weights, CI-safe)**: graph construction (adjacency shape `(K,27,27)`, symmetry, hop distances); `build_graph_features` golden test (exact `[T,27,2]`, including the 2-hand Holistic-slot mapping fixture AND the one-absent-hand fixture that exercises the post-centering re-zero); an all-zero-pose unit test asserting a finite all-zero `[27,2]` output (EPS scale guard, §3.4); node-preset/adjacency byte-identity vs the probe-captured source; `feature_kind` routing in `recognize_clip` with a fake graph model; manifest parsing for new arch names + `feature_kind`; `sync_vocabulary.py` schema test (emits `{note, signs:[{label,category,demo}]}`, `category` falls back to `"include50"`, `demo:false`); join-key normalization on sample CSV rows; CSV-header assertion (reject a header that is not byte-exactly `Category,Word,Video,FilePath`); train/test tree path-disjointness; `--official-split`-tied leakage guard (`gather_items` reads TRAIN tree only, `stratified_split` receives TRAIN items exclusively); three-way label-order test (vocabulary.json == manifest == label_map.json); warm-start acceptance guard (partial remap → fallback to random init + `warmstart_loaded_fraction == 0.0`; complete remap → accepted with fraction `>= 0.90` and conv-shape check passing).
- **Model unit**: DecoupledGCN and ST-GCN forward on random `(B,2,T,27)` → `(B,num_classes)`; ONNX parity for the new archs within atol.
- **Integration (local, needs data + MediaPipe bundle, not CI)**: `prepare_include.py --split-csv-*` over a handful of INCLUDE-50 clips → two `.npz` trees of `[T,27,2]`; one short training run; `pick_thresholds.py` on a TRAIN-derived VAL slice; a `recognize_clip` end-to-end over a real clip through the graph bundle.
- The probe (`tools/probe_openhands_assets.py`) is itself re-runnable and its report is committed.

---

## 9. Risks & mitigations

- **External facts drift** (unmaintained repo): mitigated by `tools/probe_openhands_assets.py` + committed `probe_report.json`; re-run first, drop any un-reproduced fact.
- **DecoupledGCN re-implementation fidelity**: port the three OpenHands graph files faithfully; ONNX-parity-test; ST-GCN member is a simpler fallback if DecoupledGCN underperforms.
- **Holistic-slot reconstruction** (riskiest step): specified as the §3.4 contract and pinned by a 2-hand golden fixture; `landmarks.py` docstring contradiction fixed.
- **Warm-start remap mismatch / silent partial load**: optional path, fails closed to random init; off critical path; the shim verifies pickle structure before trusting it, AND a minimum-match guard (≥90% of param tensors loaded + all graph/temporal-conv weights shape-matched, else random init) prevents a mostly-random model being mislabeled "warm-started"; `warmstart_loaded_fraction` recorded in run metadata (§4).
- **INCLUDE-50 subset may approach full-set size**: measured by `select_include50_downloads.py` and user-gated before any fetch.

---

## 10. Decision summary

1. OpenHands: **NO-GO as dependency / Lightning-checkpoint-via-package; GO on re-implementing DecoupledGCN + ST-GCN in `signtalk_core`**, optional best-effort warm-start. External facts re-runnable via `tools/probe_openhands_assets.py` (report committed; latest run all-reproduced).
2. Data: **INCLUDE-50 official split first**, user-gated subset download from Zenodo (`record/4010759`) with a **measured** size (not an estimate), computed by `select_include50_downloads.py` before the gate; split CSVs from the 47 KB `include_metadata.zip`.
3. Features: **new `[T,27,2]` graph tensor** in `graph_features.py` (single source of truth, explicit Holistic-slot contract §3.4), new golden test; 374-dim `features.py` kept until deliberate retirement.
4. Serving: **new `decoupled_gcn`/`st_gcn` manifest arch names + `feature_kind="graph"`** behind the `RecognizerBundle` seam, with the concrete forward-adapter edits in §6; `vocabulary.json` regenerated from `label_map.json`; thresholds re-tuned on a TRAIN-derived VAL slice via a graph-aware `pick_thresholds.py`.
5. Evaluation: **official INCLUDE-50 test tree only**, top-1 + top-5 + risk-coverage, with pipeline-routing leakage guards (two trees, VAL carved from TRAIN, `stratified_split`-not-on-official-path test).

---

## 11. Responses to design-review findings

Round-2 revision. Review file: `.agents/tasks/gcn-recognizer/design-review.json` (verdict CHANGES_REQUESTED; 2 HIGH, 6 MEDIUM, 4 NIT). Each finding is **addressed**, **backlogged**, or **justified**; all choices align with the original requirements (signer-independent accuracy, honest official-split evaluation, keep the app working).

- **Finding 1 (HIGH) — unverifiable external facts.** ADDRESSED. Added committed, re-runnable `tools/probe_openhands_assets.py` that regenerates every cited fact and writes `probe_report.json` with a `drift` list; ran it this round (`all_reproduced: true`, report committed at `.agents/tasks/gcn-recognizer/probe_report.json`). §2.0 states the numbers are an authoring-time snapshot that MUST be re-confirmed by the probe as the first implementation action, and any un-reproduced fact is dropped. The one fact a HEAD-only probe can't fully prove (the `.ckpt` pickle structure for the warm-start shim) is now explicitly confined to the optional `import_openhands_weights.py`, which verifies structure and fails closed (§4). Byte counts updated to the live `Content-Length` values (47,041 / 40,608,718 / 32,294,362 / 1,573,733,886).

- **Finding 2 (HIGH) — official split not actually enforced at build time.** ADDRESSED. §7.1 fully specifies the rewrite: parse both CSVs; normalized case/extension-insensitive join key on `(Category, Word-folder, Video-stem)` with `clean_label`; emit `features/{train,test}` trees + a `split` column; surface unmatched CSV rows and orphan clips as errors; §7.2 changes `train.py` to fit on TRAIN and carve VAL from TRAIN only, never touching test; §7.3 adds the pipeline-routing leakage test (not just CSV-content) and a `stratified_split`-not-on-official-path assertion. Noted `group_split` does not exist.

- **Finding 3 (MEDIUM) — forward adapter has no home.** ADDRESSED. §6 item 2 names the exact edits: `feature_kind: str = "vector"` on `RecognizerBundle`; new `build_tta_graph_views` in `tta.py` (reusing the trim logic, calling `build_graph_features`); new `_graph_model_probs_over_views` doing the `[V,T,27,2]→[V,2,T,27]` permute; `recognize_clip` branches on `feature_kind`; `RecognizerBundle` docstring contract updated for graph members.

- **Finding 4 (MEDIUM) — export_onnx is vector-only.** ADDRESSED. §4 now states it is a NEW branch: add `decoupled_gcn`/`st_gcn` to `--arch`, branch `_load` to GCN modules, build `dummy = torch.randn(1,2,seq_len,27)` (not `feature_dim()`), `dynamic_axes={"features":{0:"batch",2:"time"},"logits":{0:"batch"}}`, keep the parity assertion.

- **Finding 5 (MEDIUM) — vocabulary.json third label copy.** ADDRESSED. §6 item 3 adds `tools/sync_vocabulary.py` to regenerate `config/vocabulary.json` from `label_map.json` in index order, and a three-way order test (vocabulary.json == manifest labels == label_map.json); invariant ownership in §6 updated to include vocabulary.json.

- **Finding 6 (MEDIUM) — Holistic reconstruction only in prose + contradictory docstring.** ADDRESSED. New §3.4 gives `build_graph_features` as an explicit contract over the extended `FrameLandmarks` (full pose[33] + two hands[21]), midline from full-pose 11/12, handedness-by-midline with the existing tie-break, zero-fill + validity for an absent hand, then the 27-select and normalize; a 2-hand golden fixture pins the 27×2 layout (§3.3 step 3 / §8). The contradictory `landmarks.py` module docstring is corrected as part of this work.

- **Finding 7 (MEDIUM) — download size unknown at the gate.** ADDRESSED. §5 requires `select_include50_downloads.py` to run FIRST (needs only the 47 KB metadata) and print the category list + summed `Content-Length`; the gate presents that **measured** number, not an estimate. The "single-digit GB" prose is withdrawn and replaced with the explicit warning that whole-category zips mean the subset may approach the full set.

- **Finding 8 (MEDIUM) — pick_thresholds assumed drop-in.** ADDRESSED. §6 item 4 reconciles the real interface: `_build_members` dispatches only gru/transformer and `_fused_val_probs` runs `[1,T,F]` via `FeatureDataset`; it gets the graph branch (new archs + the shared `permute_for_graph` helper from §7.2 producing `[1,2,T,27]`) and must read a VAL slice carved from the TRAIN tree only, not `stratified_split` over the combined set. Listed among the files the `feature_kind` change touches.

- **Finding 9 (NIT) — threshold provenance misstated.** ADDRESSED. §6 item 4 now reads "the current `0.85/0.3` conservative defaults were set without a clean split and are not calibrated to the new model," and notes the leaky auto-pick was `0.5/0.0` per the config comment.

- **Finding 10 (NIT) — prepare_include flag names.** ADDRESSED. §7.1 references `--pose-model`/`--hand-model` and instructs fixing the stale `--model holistic_landmarker.task` docstring example in the script.

- **Finding 11 (NIT) — "mirrors normalization" oversells parity.** ADDRESSED. §3.2 now says it mirrors the body-relative normalization (shoulder-midpoint origin, inter-shoulder scale) and that hand-local re-normalization and velocity/presence are intentionally dropped.

- **Finding 12 (NIT) — nonexistent group_split.** ADDRESSED. §7 and §7.2 state `group_split` is unimplemented, must not be used, and separation comes from the official CSV split trees.

---

## 12. Responses to round-3 design-review findings

Round-3 revision. Review file: `.agents/tasks/gcn-recognizer/design-review.json` (verdict CHANGES_REQUESTED; 1 HIGH, 1 MEDIUM, 3 NIT), review doc `design-review.md`. Each finding is **addressed**; all choices align with the original requirements (signer-independent accuracy, honest official-split evaluation, keep the app working). I re-verified each cited file in the worktree before editing.

- **Finding 1 (HIGH) — `train.py` never got the GCN input-adapter edit.** ADDRESSED. Verified in `training/train.py`: `build_arch` dispatches only `gru`/`transformer` (raises otherwise), `--arch choices=["gru","transformer"]`, the train loop and `evaluate()` both call `model(x)` on `FeatureDataset` batches with no permute, and VAL comes from `stratified_split(items, ...)` over all gathered items. §7.2 is rewritten from prose into concrete edits that mirror §6 item 4: extend `build_arch` + `--arch choices` to `decoupled_gcn`/`st_gcn`; introduce a **single** `permute_for_graph(x) = x.permute(0,3,1,2).contiguous()` helper in a shared module, applied right after the device move in both the train loop and `evaluate()` (gated on arch/feature_kind); and the TRAIN-only VAL carve now has an attached code edit (point `gather_items` at the `features/train` tree, `stratified_split` over TRAIN items only, never touch `features/test`), plus the pipeline-routing test. §4 (ONNX dummy), §6 item 2 (`_graph_model_probs_over_views`), and §6 item 4 (pick_thresholds) were all updated to call the SAME `permute_for_graph` helper, so the `[*,T,27,2]→[*,2,T,27]` transform has exactly one definition and the silent-axis-bug risk is removed.

- **Finding 2 (MEDIUM) — `.ckpt` warm-start premise labeled "confirmed" with no committed evidence.** ADDRESSED. Verified `tools/import_openhands_weights.py` does not exist in the worktree and the committed probe makes only HEAD + two raw-file GETs (never opens the `.ckpt`). §2.1 reworded: the pickle-structure/tensor-recoverability claim is now stated as an **UNVERIFIED assumption**, "to be verified by `tools/import_openhands_weights.py`, which is NOT yet written; until it runs the warm-start path stays off and the model trains from random init." §4 now lists `import_openhands_weights.py` as an explicit, not-yet-written deliverable that is the first thing to actually open the `.ckpt` and falls back to random init if the structure differs. Low project impact (warm-start is optional and fail-closed), but the incorrect "confirmed" label is removed.

- **Finding 3 (NIT) — "mirrors existing '4,257 vs 4,292' guard" misstates the code.** ADDRESSED. Verified in `prepare_include.py`: the string is only a module-docstring anecdote; the code logs per-clip rows to `manifest.csv` with no count-mismatch check. §7.1 reworded to "add a NEW count-reconciliation guard (there is no existing count check)".

- **Finding 4 (NIT) — §3.4 step 2 misdescribes the two-hand handedness rule.** ADDRESSED. Verified in `features.py::assign_handedness`: the midline test is used only for the single-hand case; two hands are sorted purely by wrist-x (no midline). §3.4 step 2 rewritten to state the two cases exactly as implemented — single hand → wrist-x vs shoulder midline; two hands → wrist-x sort, leftmost to left block, rightmost to right block, no midline comparison.

- **Finding 5 (NIT) — probe's `exact_list_present` check is whitespace-brittle.** ADDRESSED. §3.1 adds a "Probe hardening" note: change `tools/probe_openhands_assets.py` to parse the preset list from the fetched `pose_transforms.py` into a list of ints and compare against `EXPECTED["preset_27"]` (already a Python list in the script), keeping the raw-text marker match only as a secondary signal, so trivial upstream reformatting no longer raises spurious drift.

---

## 13. Responses to round-4 design-review findings

Round-4 revision. Review file: `.agents/tasks/gcn-recognizer/design-review.json` (verdict CHANGES_REQUESTED; 4 MEDIUM, 2 NIT), review doc `design-review.md`. Each finding is **addressed**; all choices align with the original requirements (Indian Sign Language only; signer-independent accuracy on the official split as the sole success criterion; keep the whole app working; fail-closed OpenHands warm-start; surface download size before fetching gigabytes). I re-verified each cited file in the worktree before editing.

- **Finding 1 (MEDIUM) — `strict=False` warm-start can silently load near-nothing and still "succeed."** ADDRESSED. §4 adds an explicit **"Warm-start load acceptance — minimum-match guard"** block. After `load_state_dict(strict=False)`, `import_openhands_weights.py` computes `warmstart_loaded_fraction = 1 - len(missing_keys)/len(model_state_dict)` and accepts the warm-start only if **both** (a) `fraction >= 0.90` AND (b) every graph-conv + temporal-conv weight tensor matched by name and shape (names enumerated from the module tree, not substring-guessed). On failure it logs the unmatched keys, re-initializes from random init, and sets the fraction to `0.0`. The fraction (and the conv-shape pass/fail) is recorded in the run metadata so every accuracy number is honest about whether warm-start happened. §9's warm-start risk line and §8's test list (partial-remap → fallback; complete-remap → accept) are updated to match. Fail-closed and off the critical path, as required.

- **Finding 2 (MEDIUM) — `sync_vocabulary.py` output schema unspecified, risks breaking the serving fallback reader.** ADDRESSED. Verified `recognize.py::_vocab_labels` reads `data["signs"]` then `s["label"]`, and the current entries carry `{label, category, demo}`. §6 item 3 now LOCKS the exact output schema `{"note":..., "signs":[{"label":<name>,"category":<CSV Category or "include50">,"demo":false}]}`, ordered by `label_map.json` index, with `label` byte-identical to the label-map key, `category` from the §7.1 CSV join (fallback literal `"include50"` for the INCLUDE-50 run), and `demo:false` always. Added the three-way order test asserting `[s["label"]] == manifest.labels == label_map key order` (element-for-element). §8 lists the schema test and the three-way order test.

- **Finding 3 (MEDIUM) — official-split VAL carve had an ambiguous "or detect the layout" trigger.** ADDRESSED. Verified `gather_items` globs `processed_dir/features/<label>/*.npz` (zero items against the two-tree layout, or both trees if pointed loosely). §7.2 replaces the auto-detect wording with an **explicit, non-defaulting `--official-split`** flag (or `--train-tree`/`--test-tree` paths): when set, `gather_items` reads `features/train` only, `stratified_split` carves VAL over TRAIN items only, and `features/test` is untouched by `train.py`; when unset, the old single-tree random behavior stays (local experiments only). The leakage-guard test is retied to the explicit flag (assert `gather_items`/`stratified_split` see TRAIN items exclusively), not to a layout heuristic.

- **Finding 4 (MEDIUM) — zero-fill of absent/no-pose nodes collides with shoulder normalization.** ADDRESSED. §3.4 now specifies: (a) the **`_EPS = 1e-6` scale guard** in `build_graph_features` mirroring `features.py` (inter-shoulder distance `< _EPS → scale = 1.0`), killing the no-pose divide-by-zero/NaN; (b) the **no-pose frame output** defined as all-zero `[27,2]` with validity 0 (finite, reproducible); (c) the **absent-node representation decision** under the `C=2` lock — re-zero absent nodes AFTER centering so they sit at the normalized origin `(0,0)` (a single defined location, not a phantom hand), relying on the clip-level quality gate to reject too-invalid clips, rather than adding a `C=3` validity channel that would break the `C=2` lock and warm-start compatibility. A new step 6 performs the post-centering re-zero; validity is explicitly NOT a model input. The §3.3 golden fixture gains a **one-absent-hand frame**, and §8 adds the all-zero-pose finite-output unit test. Invariant ownership (EPS + re-zero → `graph_features.py`; too-many-invalid-frames → serving quality gate) is stated.

- **Finding 5 (NIT) — split-CSV source ambiguity (GitHub `include_metadata.zip` vs Zenodo `Train_Test_Split.zip`).** ADDRESSED. Verified `data/raw/include/download_data.sh` ends with `wget .../Train_Test_Split.zip` (a second on-disk split source). §5 now LOCKS the split-CSV source to the **GitHub `include_metadata.zip`** explicitly (not the Zenodo zip) and requires asserting the header is byte-exactly `Category,Word,Video,FilePath` before any join. §7.1 echoes this: `--split-csv-train/test` are the GitHub-zip CSVs, and both `prepare_include.py` and `select_include50_downloads.py` assert the exact header and abort loudly on mismatch.

- **Finding 6 (NIT) — INCLUDE-50 → INCLUDE-263 gate was qualitative.** ADDRESSED. §7.3 replaces "clearly beats / no longer collapses" with three concrete, recorded floors that must ALL hold: (1) official-test top-1 **≥ 70%** on the signer-independent split; (2) accepted-accuracy **≥ 85%** at tuned thresholds; (3) a manual webcam check showing **≥ 2 distinct accepted labels**. The absence of a trustworthy signer-independent baseline is stated explicitly (hence absolute floors rather than "beats baseline"), and all three numbers are recorded in run metadata for an auditable go/no-go.
