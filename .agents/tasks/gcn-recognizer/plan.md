# Implementation Plan — Signer-Independent ISL Recognizer via Skeleton-GCN

Source of truth: the APPROVED design at
`c:\Users\ABHIMAN KUMAR\Documents\GitHub\Signtalk-AI\.agents\tasks\gcn-recognizer\design.md`
and the committed probe facts at
`c:\Users\ABHIMAN KUMAR\Documents\GitHub\Signtalk-AI\.worktrees\gcn-recognizer\.agents\tasks\gcn-recognizer\probe_report.json`
(`all_reproduced: true`, `drift: []`).

ALL work happens in the existing worktree on branch `feat/gcn-recognizer`. Do NOT create a new worktree.

Conventions used by every item below (discovered during exploration — do not re-derive):

- Worktree root (WT): `c:\Users\ABHIMAN KUMAR\Documents\GitHub\Signtalk-AI\.worktrees\gcn-recognizer`
- Python interpreter (PY): `c:\Users\ABHIMAN KUMAR\Documents\GitHub\Signtalk-AI\.venv\Scripts\python.exe`
- Tests: `& PY -m pytest core/tests server/tests -q` (baseline today = **48 passed**).
- Lint: `& PY -m ruff check core/ server/ training/ tools/` (baseline = **All checks passed!**).
  NOTE: bare `ruff check` is RED because of legacy `backend/` + `generate_guide_pdf.py` files that are
  out of scope for this task; always lint the four package dirs, which is the HANDOVER-documented command.
- One backend at a time on port 8000, no `--reload`.
- torch 2.2.2 (CUDA available), mediapipe 0.10.14, numpy 1.26.4, Python 3.12. NO tensorflow, NO
  pytorch-lightning / hydra / omegaconf / pytorchvideo / timm / transformers added.
- `Config` (core/signtalk_core/config.py) exposes `cfg.sequence_length` (=32), `cfg.tta.{views,trim_frames}`.
- The 27-node preset and 26-edge `inward_edges` are the GRAPH-index lists in design §3.1 / §4; the probe's
  `expected_snapshot.preset_27` holds the exact preset list.

The headline success criterion is **signer-INDEPENDENT top-1 AND top-5 accuracy on the OFFICIAL
INCLUDE-50 test split** (signers unseen in training) — never a random split, never the user's own clips.

Each item leaves the codebase buildable and the existing app working (the vector GRU/Transformer path
stays intact until a deliberate retirement that is out of scope here).

---

## Phase A — Graph foundations + feature builder (pure, CI-safe)

- [ ] 1. Add the frozen 27-node graph constant + spatial-partition adjacency builder.
      Create `core/signtalk_core/graph.py` holding: `PRESET_27` (the 27 MediaPipe-Holistic indices,
      design §3.1), `INWARD_EDGES` (the 26 GRAPH-index edges, design §4), `NUM_NODES = 27`, `CENTER = 0`,
      and `build_adjacency(strategy="spatial", max_hop=1) -> np.ndarray` of shape `(K, 27, 27)` ported
      from OpenHands `graph_utils.py` (pure NumPy; normalized adjacency with the spatial partition).
      Files: `WT/core/signtalk_core/graph.py`, `WT/core/tests/test_graph.py`
      Verify: `& PY -m pytest core/tests/test_graph.py -q` — adjacency shape `(K,27,27)`, per-partition
      symmetry/row-normalization, hop distances, and byte-identity of `PRESET_27` to
      `probe_report.json.expected_snapshot.preset_27` and of the edge list length (26) all pass.

- [ ] 2. Extend the landmark struct to carry the full 33-point pose and fix the contradictory docstring.
      In `landmarks.py`: add `full_pose: np.ndarray | None` (`[33,2]`) to `FrameLandmarks` (additive —
      existing `pose_subset` untouched); in `DualLandmarker.detect_frame` keep the already-computed
      `full_pose` (currently discarded) and set it on the struct (`None`/zeros when no pose). Correct the
      module top docstring so it matches reality (separate Pose+Hand `DualLandmarker`, NOT combined
      `HolisticLandmarker`). Update `make_golden.py`/`test_features_golden.py` fixture reconstruction only
      if needed so the 374-dim golden test still passes unchanged.
      Files: `WT/core/signtalk_core/landmarks.py` (+ `WT/core/tests/golden/make_golden.py` and
      `WT/core/tests/test_features_golden.py` only if the struct change touches the fixture loader)
      Verify: `& PY -m pytest core/tests/test_features_golden.py core/tests/test_handedness.py -q` — the
      existing 374-dim golden test and handedness tests still pass (feature path untouched).

- [ ] 3. Create `build_graph_features` — the single source of truth for the `[T,27,2]` GCN tensor.
      Create `core/signtalk_core/graph_features.py` implementing the §3.4 contract: per frame compute the
      shoulder midline from full-pose indices 11/12; assign handedness exactly as `features.assign_handedness`
      does (one-hand = side-of-midline, two-hand = leftmost-wrist-to-left, MediaPipe L/R ignored); assemble a
      75×2 Holistic frame (rows 0–32 full pose, 33–53 left-assigned hand, 54–74 right-assigned hand) with a
      parallel `[75]` validity vector; apply the frozen 27-index `PoseSelect`; `CenterAndScaleNormalize`
      (shoulder-midpoint origin via graph nodes 3/4, inter-shoulder scale, `scale_factor=1`) with the SAME
      `_EPS=1e-6` guard as `features.py` (scale<EPS → 1.0); re-zero every node whose validity bit is 0 AFTER
      centering; stack frames and linear-resample to `cfg.sequence_length` → `[T,27,2]` float32.
      Files: `WT/core/signtalk_core/graph_features.py`
      Verify: `& PY -m pytest core/tests/test_graph_features.py -q` (added in item 4) — see item 4.

- [ ] 4. Pin `build_graph_features` with a golden fixture + edge-case unit tests.
      Add `core/tests/golden/make_graph_golden.py` (mirrors `make_golden.py`) producing
      `golden/input_landmarks_graph.npz` + `golden/expected_graph_features.npy`, with a deterministic clip
      containing BOTH a 2-hand frame AND a one-absent-hand frame. Add `core/tests/test_graph_features.py`
      asserting: exact `[T,27,2]` match (atol 1e-6), node count 27, dtype float32; an all-zero-pose frame
      yields a finite all-zero `[27,2]` (EPS branch); the absent-hand node sits at the normalized origin.
      Files: `WT/core/tests/golden/make_graph_golden.py`,
      `WT/core/tests/golden/input_landmarks_graph.npz`, `WT/core/tests/golden/expected_graph_features.npy`,
      `WT/core/tests/test_graph_features.py`
      Verify: `& PY -m pytest core/tests/test_graph_features.py -q` — all new tests pass.

- [ ] 5. Implement DecoupledGCN + ST-GCN modules with a shared `permute_for_graph` helper.
      Create `core/signtalk_core/models/graph_layers.py` (porting the pure-PyTorch DecoupledGCN spatial/
      temporal conv + DropGraph and the ST-GCN unit from OpenHands, no exotic deps). Create
      `core/signtalk_core/models/decoupled_gcn.py` with `DecoupledGCN(nn.Module)` + `build_decoupled_gcn(
      num_classes, checkpoint_path=None, device="cpu")` and `core/signtalk_core/models/st_gcn.py` with
      `STGCN(nn.Module)` (edge_importance_weighting=True, dropout 0.05) + `build_st_gcn(...)` — both accept
      `(N,2,T,27)` → logits `(N,num_classes)`, mirroring the `build_model`/`build_transformer` ckpt-loading
      contract. Define ONE `permute_for_graph(x) -> x.permute(0,3,1,2).contiguous()` (`[*,T,27,2]→[*,2,T,27]`)
      in a shared module (e.g. `core/signtalk_core/graph_features.py` or a small `models/graph_common.py`)
      imported by train/serve/export/pick_thresholds.
      Files: `WT/core/signtalk_core/models/graph_layers.py`,
      `WT/core/signtalk_core/models/decoupled_gcn.py`, `WT/core/signtalk_core/models/st_gcn.py`,
      shared permute module, `WT/core/tests/test_graph_models.py`
      Verify: `& PY -m pytest core/tests/test_graph_models.py -q` — both models forward on random
      `(B,2,T,27)` → `(B,num_classes)`; `permute_for_graph` maps `[B,T,27,2]→[B,2,T,27]`.

## Phase B — Training / export / serving / threshold plumbing

- [ ] 6. Teach `train.py` + `dataset.py` the GCN archs, the shared permute, and the `--official-split` flag.
      In `training/train.py`: add `decoupled_gcn`/`st_gcn` to `build_arch` and to `--arch` choices (calling
      the same factory functions serving uses); apply `permute_for_graph` right after `x.to(device)` in BOTH
      the train loop and `evaluate()`, gated on `arch in {"decoupled_gcn","st_gcn"}`; add a non-defaulting
      `--official-split` flag — when set, point `gather_items` at `features/train` ONLY, carve VAL out of
      TRAIN via per-class `stratified_split`, and never touch `features/test`; when unset keep current
      single-tree random behavior. In `dataset.py`: correct the docstring noting `group_split` is
      unimplemented and the official CSV split supersedes it (loader is already shape-agnostic).
      Files: `WT/training/train.py`, `WT/training/dataset.py`, `WT/core/tests/test_official_split_guard.py`
      Verify: `& PY -m pytest core/tests/test_official_split_guard.py core/tests/test_graph_models.py -q` —
      with `--official-split` set, `gather_items` is invoked against the TRAIN tree only and `stratified_split`
      receives TRAIN-tree items exclusively; `features/test` paths never appear in the fit/VAL set.

- [ ] 7. Add the GCN forward adapter to serving (`recognizer.py` + `recognize.py` + `tta.py`).
      In `server/app/recognizer.py::load_bundle`: dispatch `"decoupled_gcn"`→`build_decoupled_gcn`,
      `"st_gcn"`→`build_st_gcn`; read optional top-level `"feature_kind"` (default `"vector"`) and pass to
      the bundle. In `recognize.py`: add `feature_kind: str = "vector"` to `RecognizerBundle` (+ docstring for
      `[B,2,T,27]` graph members); add `_graph_model_probs_over_views` that stacks `[V,T,27,2]`, applies the
      shared `permute_for_graph` → `[V,2,T,27]`, then `model(batch)` (temperature/top-1 identical to vector
      path); branch `recognize_clip` on `bundle.feature_kind`. In `tta.py`: add `build_tta_graph_views` reusing
      the trim logic but calling `build_graph_features`. Fallback (no bundle) stays vector.
      Files: `WT/server/app/recognizer.py`, `WT/core/signtalk_core/recognize.py`,
      `WT/core/signtalk_core/tta.py`, `WT/server/tests/test_recognize_graph.py`
      Verify: `& PY -m pytest core/tests server/tests -q` — new graph-routing test (fake graph model via a
      `feature_kind:"graph"` bundle produces a valid 6.1 dict) passes AND all 48 existing tests still pass.

- [ ] 8. Add the GCN ONNX export branch to `export_onnx.py`.
      Add `decoupled_gcn`/`st_gcn` to `--arch`; branch `_load` to the GCN factories (do NOT call
      `feature_dim()` for GCN); build `dummy = permute_for_graph(torch.randn(1, args.seq_len, 27, 2))`
      (= `[1,2,seq_len,27]`); set `dynamic_axes={"features":{0:"batch",2:"time"},"logits":{0:"batch"}}`;
      keep the existing PyTorch↔ONNX parity assertion (non-zero exit on mismatch).
      Files: `WT/training/export_onnx.py`, `WT/core/tests/test_onnx_parity_gcn.py`
      Verify: `& PY -m pytest core/tests/test_onnx_parity_gcn.py -q` — exporting a tiny random-init GCN and
      running ONNX Runtime matches PyTorch within atol (onnxruntime 1.18.0 is installed per requirements).

- [ ] 9. Add graph plumbing to `tools/pick_thresholds.py`.
      Add `decoupled_gcn`/`st_gcn` to `_build_members`; in `_fused_val_probs` feed graph `.npz` as
      `[1,2,T,27]` via the shared `permute_for_graph`; on the official-split path read the VAL slice carved
      from the TRAIN tree only (never `stratified_split` over a combined train+test set). Keep write-config
      behavior.
      Files: `WT/tools/pick_thresholds.py`, `WT/core/tests/test_pick_thresholds_graph.py`
      Verify: `& PY -m pytest core/tests/test_pick_thresholds_graph.py -q` — a fake graph member runs through
      `_fused_val_probs` with `[1,2,T,27]` and returns fused probs of the right shape.

## Phase C — Data pipeline, vocabulary, optional warm-start

- [ ] 10. Rewrite `prepare_include.py` for the official two-tree split using `build_graph_features`.
      Add `--split-csv-train` / `--split-csv-test` (CSVs from the GitHub `include_metadata.zip`). Assert the
      CSV header is byte-exactly `Category,Word,Video,FilePath` before any join (abort loudly otherwise).
      Build a case/extension-insensitive join key `(Category, Word-folder, Video-stem)` with `clean_label`;
      emit `features/train/<label>/<stem>.npz` and `features/test/<label>/<stem>.npz` built with
      `build_graph_features`, plus a `split` column in `manifest.csv`; add a count-reconciliation guard
      (unmatched CSV rows + orphan clips logged; non-zero count fails the run). Build `label_map.json` from the
      union of labels across both trees. Fix the stale `--model holistic_landmarker.task` docstring example.
      Files: `WT/training/prepare_include.py`, `WT/core/tests/test_prepare_include_join.py`
      Verify: `& PY -m pytest core/tests/test_prepare_include_join.py -q` — header assertion rejects a non-
      matching header; join-key normalization matches `"Animals/4. Bird/MVI_2987.MOV"`-style rows to on-disk
      stems case/extension-insensitively; train/test output trees are path-disjoint.

- [ ] 11. Add `tools/sync_vocabulary.py` + the three-way label-order test.
      Create `tools/sync_vocabulary.py` that regenerates `config/vocabulary.json` from a trained
      `label_map.json` in class-index order, emitting EXACTLY `{note, signs:[{label,category,demo}]}` with
      `label` byte-identical to the label_map key, `category` from the split-CSV join or literal `"include50"`
      fallback (never null), `demo:false`. Add a test asserting
      `[s["label"] for s in vocabulary.json["signs"]] == manifest.labels == list(label_map keys in index order)`.
      Files: `WT/tools/sync_vocabulary.py`, `WT/core/tests/test_sync_vocabulary.py`
      Verify: `& PY -m pytest core/tests/test_sync_vocabulary.py -q` — schema + three-way element-for-element
      order equality pass; `_vocab_labels` in `recognize.py` can read the emitted file.

- [ ] 12. Add the fail-closed optional warm-start importer `tools/import_openhands_weights.py`.
      Download `include_slgcn.zip`; extract the `.ckpt` with a `pytorch_lightning.*`-stubbing unpickler shim
      (NO Lightning install); first verify the `.ckpt` is a torch zip whose top-level pickle references only
      stubbable `pytorch_lightning.*` modules — if not, fall back to random init and never block. Remap keys to
      `DecoupledGCN`, `load_state_dict(strict=False)`, then apply the acceptance gate: accept only if
      `warmstart_loaded_fraction >= 0.90` AND every graph-conv + temporal-conv weight tensor matched by name and
      shape; otherwise log, discard, re-init random, set fraction 0.0. Record `warmstart_loaded_fraction` + the
      conv-shape check result in run metadata. 263-class only; gated behind a flag; off the critical path.
      Files: `WT/tools/import_openhands_weights.py`, `WT/core/tests/test_warmstart_guard.py`
      Verify: `& PY -m pytest core/tests/test_warmstart_guard.py -q` — a deliberately-partial remap triggers
      fallback-to-random (`warmstart_loaded_fraction == 0.0`); a complete remap is accepted (fraction >= 0.90,
      conv-shape check passes). No network in the test (shim/gate exercised with synthetic state_dicts).

## Phase D — Acquire INCLUDE-50, train, evaluate (user-gated download)

- [ ] 13. Add `tools/select_include50_downloads.py` and SURFACE the measured download size to the user.
      Needs only the 47 KB `include_metadata.zip` (no video). Parse `train_include50.csv` + `test_include50.csv`
      (same header assertion), map each referenced `Category` to its Zenodo whole-category zip set, HEAD each zip
      for `Content-Length`, and print the category list + summed byte total (and the "wasted" fraction vs clips
      actually needed). Then STOP and present option (A) measured INCLUDE-50 category-zip set [recommended] vs
      (B) full INCLUDE-263 (~56 GB) to the user and WAIT for approval before any multi-GB fetch.
      Files: `WT/tools/select_include50_downloads.py`, `WT/core/tests/test_select_downloads.py`
      Verify: `& PY -m pytest core/tests/test_select_downloads.py -q` — category→zip mapping + byte-summation
      over a fixture CSV, header assertion. Then run `& PY tools/select_include50_downloads.py ...` and report
      the measured total to the user as the download gate (do not download until approved).

- [ ] 14. After approval: download INCLUDE-50, run `prepare_include.py --official-split`, TRAIN the GCN.
      Download the approved Zenodo category zips, extract, keep only clips in the two CSVs. Run
      `prepare_include.py --split-csv-train ... --split-csv-test ...` → two `[T,27,2]` feature trees. Train the
      DecoupledGCN (and ST-GCN as the second ensemble member) with `train.py --arch decoupled_gcn
      --official-split ...` (train-time augmentation per design §3.2; VAL carved from TRAIN only). Optionally run
      the warm-start importer (item 12) first; record `warmstart_loaded_fraction`. Then
      `pick_thresholds.py --official-split` on a TRAIN-derived VAL slice and `sync_vocabulary.py`.
      Files: `WT/data/processed/include50/**` (gitignored), `WT/runs/**` (gitignored),
      `WT/models/*.onnx` + `WT/models/ensemble.json` + `WT/config/vocabulary.json`
      Verify: training completes; `& PY -m pytest core/tests server/tests -q` still green; one backend boots on
      port 8000 and logs `loaded ensemble` with `feature_kind=graph`.

- [ ] 15. EVALUATE on the OFFICIAL signer-independent INCLUDE-50 test split and record the gate numbers.
      Load `features/test` (= `test_include50.csv`) EXACTLY ONCE at the end; report top-1 and top-5, plus
      accepted-accuracy and coverage at the tuned thresholds. Assert the leakage guards (train/test trees
      path-disjoint; `stratified_split` not on the official path). Record all gate numbers next to
      `warmstart_loaded_fraction` in run metadata. Gate to proceed to INCLUDE-263: official-test top-1 ≥ 70%,
      accepted-accuracy ≥ 85%, and a manual webcam check showing ≥ 2 distinct accepted labels (refuting the
      single-label collapse). If any floor is missed, diagnose the pipeline — do NOT download INCLUDE-263.
      Files: evaluation script/output under `WT/runs/**`, run-metadata JSON
      Verify: the official-test top-1/top-5 are produced from `features/test` only; `& PY -m pytest
      core/tests server/tests -q` green; `& PY -m ruff check core/ server/ training/ tools/` passes.

---

## Notes / assumptions

- Lint scope: the task says `ruff check`; exploration shows bare `ruff check` is RED only because of
  out-of-scope legacy `backend/` and `generate_guide_pdf.py` files. All verification uses the
  HANDOVER-documented scoped form `ruff check core/ server/ training/ tools/`, which is green at baseline.
- The 374-dim `features.py` + its golden test stay UNCHANGED and passing throughout; retiring them is a
  deliberate later commit, explicitly out of scope for this task.
- Items 1–13 are code+unit-test only and run fully in CI (no data, no network, no GPU). Items 14–15 need the
  downloaded dataset, MediaPipe bundle, and GPU, and are gated on the user approving the measured download.
- `permute_for_graph` must have exactly ONE definition shared by train/serve/export/pick_thresholds so the
  axis order can never drift between paths.
