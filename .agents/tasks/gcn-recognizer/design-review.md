# Design Review — Signer-Independent ISL Recognizer via Skeleton GCN (round 5)

Reviewer: design-review subagent (fresh read, no authoring context).
Design under review: `.agents/tasks/gcn-recognizer/design.md` (status: revised, round 4).
Method: read the design in full, then verified every claim it makes about current source state against the actual files in the worktree `c:\Users\ABHIMAN KUMAR\Documents\GitHub\Signtalk-AI\.worktrees\gcn-recognizer` (and the shared workspace copy of `download_data.sh`). No code was written and no training/tests were run, per the step instructions.

This is a mature design that has been through four review rounds. This pass re-checks only whether the 6 findings recorded in the prior `design-review.json` (round 4: 4 MEDIUM, 2 NIT) are GENUINELY closed in the revised text, verifying each against the actual worktree source rather than taking the design's word.

Result: all 6 findings are genuinely closed, and close reading surfaced no new HIGH or MEDIUM issue. Verdict: **APPROVED**.

---

## Verified Assumptions

Each was checked against the actual file; the design describes the current source state correctly.

1. **`features.py` has the exact EPS scale guard the design mirrors.** Source: `_EPS = 1e-6`; `scale = ||pose[_LEFT_SHOULDER] - pose[_RIGHT_SHOULDER]||`; `if scale < _EPS: scale = 1.0`; `_LEFT_SHOULDER = 1`, `_RIGHT_SHOULDER = 2` (subset-local indices = MediaPipe 11/12). The hand-local path also guards `if palm < _EPS: palm = 1.0`. Exactly matches the §3.4 Finding-4 mirror claim.
2. **`features.py::assign_handedness` implements the two-case rule the design describes.** Single hand → `wrist_x < midline` decides left/right; two hands → `hands.sort(key wrist_x)`, leftmost→left, rightmost→right, no midline comparison (the sort also resolves the same-side tie deterministically). Matches §3.4 step 2 verbatim.
3. **`recognize.py::_vocab_labels`** reads `signs = data["signs"] if isinstance(data, dict) else data; return [s["label"] for s in signs]`, and `recognize_clip` falls back to `_vocab_labels(root)` when no bundle. Matches §6 item 3 / Finding 2.
4. **`config/vocabulary.json`** is `{"note": ..., "signs": [{"label", "category", "demo"}, ...]}` (9 Greetings entries, `category: "greetings"`, `demo: true`), with the note "Order MUST match the class-index order in the trained label_map.json". Confirms the schema the design locks and that generated INCLUDE-50 entries must keep `{label, category, demo}` (design sets `demo: false`, `category` from CSV or `"include50"`).
5. **`training/train.py`** — `build_arch` dispatches only `gru`/`transformer`; `--arch choices=["gru","transformer"]`; `items = gather_items(args.processed_dir, label_map)` then `stratified_split(items, args.val_frac, args.seed)` over all gathered items; no permute. Confirms the §7.2 / Finding-3 blockers are real and the official-split flag is a genuine addition.
6. **`tools/pick_thresholds.py`** — `_build_members` dispatches only `gru`/`transformer` (`else: raise ValueError`); loop does `xb = x.unsqueeze(0)` → `[1, T, F]`; VAL via `stratified_split(items, ...)` over all gathered items. Matches §6 item 4.
7. **`training/export_onnx.py`** — `--arch choices=["gru","transformer"]`; `f = feature_dim(); dummy = torch.randn(1, args.seq_len, f)`; `input_names=["features"]`; `dynamic_axes={"features": {0: "batch", 1: "time"}, "logits": {0: "batch"}}`. Confirms export is vector-only, so §4's "NEW branch" edits are correctly scoped.
8. **`data/raw/include/download_data.sh`** fetches whole-category Zenodo zips (`https://zenodo.org/record/4010759/files/<Category>_${i}of<n>.zip`, incl. `Greetings_${i}of2.zip`) and ends with `wget https://zenodo.org/record/4010759/files/Train_Test_Split.zip`. Confirms the dual-split-source hazard that Finding 5 locks down (Zenodo ships a `Train_Test_Split.zip` while the design sources CSVs from GitHub `include_metadata.zip`).
9. **The committed probe report is consistent.** `.agents/tasks/gcn-recognizer/probe_report.json` shows `all_reproduced: true`, `drift: []`, the three release-asset sizes (47,041 / 40,608,718 / 32,294,362), Zenodo Greetings 1,573,733,886, PyPI `is_ai4bharat_slr: false`, `encoder_type: decoupled-gcn` + `fc` decoder, and the 27-int `preset_27` list `[0,2,5,11,12,13,14, 33,37,38,41,42,45,46,49,50,53, 54,58,59,62,63,66,67,70,71,74]`. The §2.1/§3.1/§5 numbers match the report.

## Unverified / Wrong Assumptions

1. **The `.ckpt` pickle-structure / tensor-recoverability premise remains UNVERIFIED** — the design states this explicitly (§2.1/§4), gates it behind the optional, not-yet-written, fail-closed `tools/import_openhands_weights.py`, and keeps the warm-start off the critical path. Correctly labeled; not a finding.
2. **Published INCLUDE-50 GCN accuracy range (§7.3 "~90%+")** is an external-literature claim not verifiable from this repo. It is used only as a soft sanity target; the actual gate (Finding 6) now uses absolute floors (70% / 85% / ≥2 labels), so the unverifiable number drives no decision. Not a finding.
3. **`download_data.sh` lives in the shared workspace, not under the worktree path.** The design cites it as a repo file; it resolves to `c:\...\Signtalk-AI\data\raw\include\download_data.sh` (an open editor file), which is the same working tree. Its content matches the design's quotation. Not a finding.

---

## Re-verification of the 6 prior findings

### Finding 1 (MEDIUM, prior) — strict=False warm-start can silently load near-nothing → CLOSED
§4 adds a "Warm-start load acceptance — minimum-match guard" block: after `load_state_dict(strict=False)`, compute `warmstart_loaded_fraction = 1 - len(missing_keys)/len(model_state_dict)`; accept only if (a) `>= 0.90` AND (b) every graph-conv + temporal-conv weight tensor matched by name AND shape (names enumerated from the module tree, not substring-guessed); otherwise log, discard, re-init random, set fraction `0.0`. The fraction and the conv-shape pass/fail are recorded in run metadata and echoed to stdout, and §8 lists the unit test (partial remap → fallback + fraction 0.0; complete remap → accept at ≥0.90). This matches the requested fix exactly (≥90% match-ratio guard + fallback + `warmstart_loaded_fraction` in run metadata). Genuinely closed.

### Finding 2 (MEDIUM, prior) — sync_vocabulary.py output schema unspecified → CLOSED
§6 item 3 LOCKS the output to the existing reader's schema: `{"note": ..., "signs": [{"label": <name>, "category": <CSV Category or "include50">, "demo": false}, ...]}`, ordered strictly by `label_map.json` class index, `label` byte-identical to the label-map key, `category` never null (falls back to literal `"include50"`), `demo` always `false`. The three-way order test asserts `[s["label"] for s in signs] == manifest.labels == list(label_map keys in index order)` element-for-element, and §8 lists both the schema test and the three-way order test. Verified against the real reader (`data["signs"]` → `s["label"]`) and the real current entry shape (`{label, category, demo}`). Matches the requested fix. Genuinely closed.

### Finding 3 (MEDIUM, prior) — official-split VAL carve had no explicit trigger → CLOSED
§7.2 removes the ambiguous "or detect the layout" wording and replaces it with a required-when-official, **non-defaulting `--official-split`** flag (equivalently `--train-tree`/`--test-tree`). When set: `gather_items` reads `features/train` only (MUST NOT glob the parent or `features/test`); `stratified_split` carves VAL over TRAIN items only and is never called on a combined set; `features/test` is untouched by `train.py` and read exactly once by the §7.3 eval step. When unset, the old single-tree random behavior remains for throwaway local runs only. The leakage-guard test asserts, keyed off the explicit flag (not a layout heuristic), that `gather_items`/`stratified_split` see TRAIN items exclusively and `features/test` never enters the fit/VAL set. Matches the requested fix. Genuinely closed.

### Finding 4 (MEDIUM, prior) — zero-fill vs shoulder normalization → CLOSED
§3.4 specifies all four pieces the fix required: (a) the explicit `_EPS = 1e-6` scale guard in `build_graph_features` (inter-shoulder distance `< _EPS → scale = 1.0`), mirroring `features.py` — verified that `features.py` actually carries `_EPS = 1e-6` and that guard; (b) the no-pose frame output defined as an all-zero `[27,2]` with validity 0 (finite, reproducible, no NaN); (c) the absent-node representation DECISION under the `C=2` lock — re-zero every validity-0 node AFTER `CenterAndScaleNormalize` (new step 6) so absent nodes sit at the normalized origin `(0,0)` rather than becoming a phantom `(-mid/scale)` position, explicitly rejecting a `C=3` validity channel because it would break the `C=2` lock and warm-start compatibility; validity is explicitly NOT a model input and is consumed only by the clip-level quality gate; (d) the §3.3 golden fixture gains a one-absent-hand frame, plus §8 adds an all-zero-pose finite-output unit test. Invariant ownership (EPS + re-zero → `graph_features.py`; too-many-invalid-frames → serving quality gate) is stated. Matches the requested fix. Genuinely closed.

### Finding 5 (NIT, prior) — split-CSV source ambiguity → CLOSED
§5 LOCKS the split-CSV source to the GitHub `include_metadata.zip` (47,041 B) explicitly and states the source is NOT the Zenodo `Train_Test_Split.zip`, noting the repo's `download_data.sh` ends with `wget .../Train_Test_Split.zip` (verified present) so a second on-disk source exists and mixing would under-match. §7.1 requires both `prepare_include.py` and `select_include50_downloads.py` to assert the CSV header is byte-exactly `Category,Word,Video,FilePath` (in that order) before any join and abort loudly on mismatch. Matches the requested fix. Genuinely closed.

### Finding 6 (NIT, prior) — gate numbers qualitative → CLOSED
§7.3 replaces "clearly beats / no longer collapses" with three concrete floors that ALL must hold: (1) official-test top-1 **≥ 70%** on `features/test`; (2) accepted-accuracy **≥ 85%** at the tuned thresholds; (3) a manual webcam check showing **≥ 2 distinct accepted labels**. The absence of a trustworthy signer-independent baseline is stated (hence absolute floors, not "beats baseline"), and all three numbers are recorded in run metadata for an auditable go/no-go. Matches the requested fix. Genuinely closed.

---

## Findings

No HIGH, MEDIUM, or NIT findings. All 6 prior findings are genuinely closed and verified against source; close reading surfaced no new issue of any severity.

---

## Verdict

Count: 0 HIGH, 0 MEDIUM, 0 NIT. HIGH + MEDIUM = 0 → **APPROVED**.

Every claim the design makes about current source state held under verification: the `_EPS`/`scale < _EPS` guard and the two-case handedness rule in `features.py`, the `_vocab_labels` reader schema and the current `vocabulary.json` shape, the gru/transformer-only dispatch and all-items `stratified_split` in `train.py`/`pick_thresholds.py`/`export_onnx.py`, the Zenodo `Train_Test_Split.zip` line in `download_data.sh`, and the committed `probe_report.json` numbers. The four MEDIUM fixes (warm-start minimum-match guard, locked `vocabulary.json` schema, explicit non-defaulting `--official-split` flag, and the EPS/no-pose/absent-node normalization contract) and the two NIT fixes (GitHub `include_metadata.zip` source + exact header assertion, and the three concrete gate floors) are all present, specific, and testable. The design is ready for implementation.
