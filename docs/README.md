# SignTalk AI — Documentation

This folder is the structured documentation set for **SignTalk AI**, a final-year
B.E. capstone (RNS Institute of Technology, ISE, Batch #24).

The single source of truth for vision, contracts and phases is
[`../PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md) (v3, accuracy-first). These docs
reorganise that document into focused files and add an honest account of what is
**currently built** versus what the v3 plan **targets**.

> If any doc here disagrees with `PROJECT_CONTEXT.md`, that file wins unless a
> human decides otherwise (per Section 0.1 of the context document).

## Documents

| File | What it covers |
|---|---|
| [VISION.md](VISION.md) | The problem, the one-sentence vision, scope, non-goals, and the honest truth about "100% accuracy". |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Target v3 architecture (browser + server), the recognition pipeline, interface contracts, and the current (v2 reference) architecture that exists in the repo today. |
| [TECH_STACK.md](TECH_STACK.md) | The target v3 stack and the stack actually present in the code, side by side. |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | Build phases, gates, the minimum path, risks, and the evaluation plan. |
| [PROGRESS.md](PROGRESS.md) | Honest snapshot: what exists in the repo now, the v2-to-v3 gap, and the open questions for the team. |

## The one thing to understand first

There are **two SignTalk AIs** in this repository:

1. **v2 (reference codebase, what the code is today):** MoveNet 17-keypoint pose
   + 30-frame BiLSTM, streaming over a WebSocket, with Gemini 2.0 / Flan-T5,
   DeepFace, Whisper, gTTS/Coqui and Firebase. This is a working-shaped skeleton
   from a previous team.

2. **v3 (the target in `PROJECT_CONTEXT.md`, not yet built):** an
   accuracy-first redesign — full video to the server, MediaPipe Holistic
   landmarks, an **ensemble** of models with calibration/fusion, test-time
   augmentation, personal **enrollment**, and an **accept / confirm / reject**
   decision engine. Record-then-recognise whole clips instead of streaming.

`PROJECT_CONTEXT.md` is explicit that v3 **supersedes** v2 and that the existing
code is **reference only** — reuse ideas, not structure. These docs keep the two
clearly separated so nobody mistakes the current code for the v3 target.
