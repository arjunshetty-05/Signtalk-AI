================================================================================
SIGNTALK AI -- AGENT CONTEXT, VISION & BUILD PLAN
VERSION 3 : ACCURACY-FIRST ARCHITECTURE  (supersedes v2)
================================================================================
Project type : Final-year B.E. capstone (RNS Institute of Technology, ISE,
               Batch #24). College project -- not a business.
Team         : Arjun B Shetty, Khushi Ramesh, Mahin S Kunder, Sathya Shriya G N
Guide        : Umesh M
Document date: October 2026
Audience     : an AI coding agent working inside an IDE, plus the human team.
Reading time : about 25 minutes. Read ALL of it before writing code. Sections
               marked "WHY" explain the reasoning so you can make good
               decisions when the plan does not cover a situation.

KEY CHANGE FROM v2
  The owner's top priority is the highest possible ACCURACY of the app, and
  the owner does not care about video privacy. v2 kept video on the device
  (privacy-first, landmarks only). v3 does the opposite: it sends FULL VIDEO
  to a server so we can use the heaviest, most accurate models, and it adds a
  set of accuracy techniques (ensembles, personal enrollment, abstain/confirm,
  context constraints). See Sections 1 and 3.


================================================================================
SECTION 0.  HOW THE AGENT MUST USE THIS DOCUMENT
================================================================================
0.1  This file is the source of truth for vision, architecture, contracts and
     phases. If code disagrees with this file, this file wins unless a human
     says otherwise.

0.2  A reference codebase from a previous team exists (FastAPI + React +
     Flutter, MoveNet 17 keypoints + BiLSTM, Gemini/Flan-T5, DeepFace, Whisper,
     Firebase). It is REFERENCE ONLY. Reuse ideas, not structure. Its "locked
     contracts" are void; Section 6 replaces them.

0.3  Build a WALKING SKELETON first (Phase 1): the thinnest end-to-end path
     (record a clip -> get a word on screen), even with a poor model. Then
     improve accuracy step by step, measuring after every change. Never
     optimise something you have not measured.

0.4  Gates: do not start a phase until the previous gate passes. A gate passes
     only when you have RUN the check and written the real command and output
     into docs/PROGRESS.md. Never claim a number you did not measure.

0.5  Label anything uncertain [VERIFY]. External facts (model names, library
     APIs, dataset counts, licenses) were researched in Oct 2026 and go stale.

0.6  Ask the human before: deleting data/models, changing a contract in
     Section 6, adding a paid service, or committing anything key-like.

0.7  HONESTY RULE (most important): the demo and paper must never overstate.
     "100%" is only ever claimed for a precisely defined closed demo scenario
     with measured trials (Section 1). Anything outside it is reported with
     its real number.


================================================================================
SECTION 1.  THE TRUTH ABOUT "100% ACCURACY" (read this twice)
================================================================================
1.1  Can open sign recognition reach 100%?
     No. No published system recognises open-vocabulary sign language at 100%.
     For reference (all signer-independent, published numbers):
       - INCLUDE paper (the dataset we use): best model 94.5% on a 50-sign
         subset, 85.6% on all 263 signs.
       - On controlled lab datasets with many repetitions and a fixed
         background (Chinese SLR500), published top-1 is about 98-99%.
       - Ensembles and RGB+pose fusion are consistently the strongest
         approaches in the literature (Section 4).
     So promising "100% on any sign, any person, any camera" would be false.

1.2  What we CAN engineer: "DEMO-100"
     A closed set of 10-20 signs, performed by enrolled signers (the team),
     in a controlled setup, where the system is designed so that EVERY WORD
     THAT IS SHOWN OR SPOKEN IS CORRECT. We get there by stacking techniques
     (Section 3) and by letting the system ABSTAIN or ASK when unsure instead
     of guessing. This is honest, measurable, and is also good engineering
     (it is how safety-relevant systems are built).

1.3  Four metrics (always report all four, never only the flattering one)
     A. RAW ACCURACY       : fraction of attempts where the model's top-1 label
                             is correct (what most papers report).
     B. ACCEPTED ACCURACY  : among attempts the system outputs automatically
                             (no question asked), fraction correct.
                             TARGET: 100% on the demo test sessions.
     C. COVERAGE           : fraction of attempts that are accepted
                             automatically. TARGET: >= 90%.
                             (The rest become "confirm" or "try again".)
     D. FINAL-OUTPUT ACCURACY: after the user taps to confirm in the "confirm"
                             case. Correct by construction unless the user
                             taps the wrong chip.
     WHY: you can push B to 100% simply by being cautious, but then C falls.
     Showing B and C together (a "risk-coverage" curve) is the honest way to
     present it and is a respectable paper result.

1.4  You cannot prove 100% with a finite test. The "rule of three":
     if you observe ZERO errors in n independent trials, the 95% upper bound
     on the true error rate is about 3/n.
         n = 30    -> error could still be up to ~10%
         n = 100   -> up to ~3%
         n = 300   -> up to ~1%
         n = 1000  -> up to ~0.3%
     So a claim like "0 errors in 300 trials" supports "accuracy >= 99% with
     95% confidence", NOT "100%". Say it that way in the paper and in the
     demo. A small test (e.g. 20 trials) proves almost nothing.

1.5  Three accuracy tiers (expectations, NOT promises; measure everything)
     TIER 1  DEMO-100   : 10-20 signs, enrolled signers, controlled setup,
                          abstain/confirm on. Expect to reach accepted
                          accuracy near 100% if the stack in Section 3 is
                          built well. This is the headline.
     TIER 2  50 SIGNS, UNSEEN SIGNERS : the honest generalisation number.
                          Expect something like the high-80s to mid-90s (the
                          INCLUDE paper's 94.5% is the reference point).
     TIER 3  ALL 263 SIGNS: reference-level, about 85% in the literature.
                          Not a goal; report only if time permits.

1.6  The #1 way to fake accuracy by accident (DO NOT DO THIS)
     DATA LEAKAGE. If the test clips come from the same recording session as
     training clips (e.g. you split frames or near-identical repetitions
     randomly), accuracy looks great and means nothing. Rule: test data must
     come from a DIFFERENT SESSION (different day/time/lighting/position) than
     training data. For enrolled signers: train on sessions 1-2, test on
     session 3, rotate. For signer-independent numbers: hold out whole signers.


================================================================================
SECTION 2.  PROJECT VISION AND SCOPE
================================================================================
2.1  Problem
     Over 466 million people live with disabling hearing loss. Most hearing
     people cannot sign; typing back and forth is slow and awkward.

2.2  Vision (one sentence)
     A web app where a signer faces a webcam and the system speaks and
     subtitles what they signed (English / Hindi / Kannada) with very high
     reliability, while the hearing person's speech appears as live captions.

2.3  Direction A -- Sign -> Voice (primary)
     webcam -> record sign clip -> server recognises the sign (accuracy stack)
     -> words are collected -> sentence is built -> translated -> subtitle +
     speech.

2.4  Direction B -- Voice -> Subtitles (secondary)
     microphone -> speech-to-text -> live captions + conversation log.
     Stretch: show a sign picture/video for recognised vocabulary words.

2.5  Honest scope statement (use everywhere, including the paper)
     "SignTalk AI performs isolated Indian Sign Language word recognition on a
     curated vocabulary. Recognised words are accumulated and a language model
     reconstructs a grammatical sentence. It is not a continuous
     sign-language translation system."

2.6  Non-goals
     - Continuous ISL translation. - Fingerspelling (unless everything else is
     done). - Native mobile app (a responsive web app is enough). - All 263
     INCLUDE signs. - Special hardware (gloves, depth cameras).

2.7  Privacy decision (recorded so nobody re-debates it)
     The owner explicitly does not need video privacy for this college
     project. Video is sent to our own server and may be stored for training
     and evaluation. Required minimum: tell participants (a one-line consent
     notice in the app), do not publish identifiable videos without consent,
     and keep the data folder out of git. Everything else about privacy is
     out of scope.


================================================================================
SECTION 3.  THE ACCURACY STACK (12 levers, with reasoning)
================================================================================
Each lever has: WHAT, WHY it helps, HOW (short), COST/RISK. Apply them in
order; each should be measured on its own (ablation) so we know what helped.

L1  ENGINEER THE VOCABULARY FOR DISTINCTNESS
    WHAT : choose the 10-20 demo signs so no two are easily confused.
    WHY  : most errors come from a few look-alike pairs. Removing the pairs
           removes the errors.
    HOW  : start with ~30 useful candidates; train; read the confusion matrix;
           for any pair confused > 2% of the time, drop one of them. Prefer
           signs that differ in handshape, location or motion. Note: INCLUDE
           merges some synonyms into one class (e.g. "big large",
           "small little") -- treat those as single signs.
    COST : smaller vocabulary. Acceptable; scope is already a closed demo set.

L2  CONTROLLED CAPTURE + PRE-FLIGHT CHECK
    WHAT : a fixed, good setup, and the app refuses to record until it is good.
    WHY  : the biggest hidden accuracy killer is bad input (hands out of frame,
           dark room, motion blur), not the model.
    HOW  : on-screen checklist that turns green: both hands + shoulders in
           frame, brightness above a threshold, distance in range, plain
           background. Practical rules: front light, plain background, solid
           dark long sleeves (skin contrasts better for hand detection), camera
           at chest height, about 1 m away, 720p or 1080p at 30 FPS.
    COST : a little setup time before the demo. Worth it.

L3  RECORD-THEN-RECOGNISE (WHOLE CLIP), NOT STREAMING WINDOWS
    WHAT : record one complete sign, send the whole clip, classify it once.
    WHY  : the previous team found a continuously sliding 30-frame window saw
           partial motion and produced flapping wrong labels, while the SAME
           clip classified correctly when processed whole. Training data is
           whole clips, so inference should be too.
    HOW  : the browser starts/stops recording automatically when hands rise /
           return to rest, or by holding the Space key (push-to-sign, always
           available as a fallback).
    COST : the word appears ~2-5 seconds after the sign ends instead of
           "instantly". Accuracy matters more than speed here.

L4  FULL-QUALITY VIDEO TO THE SERVER (privacy waived)
    WHAT : send full-resolution, full-frame-rate clips; run the most accurate
           landmark settings and models on the server.
    WHY  : v2 compressed everything to run in the browser. Server-side we can
           use a heavier landmark model, all frames (not every second one),
           and RGB-based models that need real pixels.
    HOW  : upload each clip (WebM/MP4 or a frame batch) to POST /api/recognize.
    COST : bigger uploads (fine on localhost/LAN); needs a decent server
           machine (GPU helps but is not required).

L5  RICH FEATURES, INCLUDING HAND DETAIL AND RGB CROPS
    WHAT : feed the models more information than body-point positions.
    WHY  : handshape (finger positions) carries much of a sign's meaning;
           MoveNet's 17 points have wrists but no fingers.
    HOW  : MediaPipe Tasks Holistic (or Pose + Hand landmarkers): 21 landmarks
           per hand + upper-body pose, 2D only (a study found adding depth hurt
           results). Add hand-local coordinates, velocity, hand-presence
           masks. For the RGB branch, crop the hands/face/upper body using the
           landmarks so the model sees fine detail.
    COST : more code; handled by shared feature module (Section 5.4).

L6  START FROM PRETRAINED MODELS
    WHAT : do not train from scratch.
    WHY  : pretraining helps most when data is small (our case): AI4Bharat's
           OpenHands paper shows pretraining improves fine-tuning, especially
           in low-resource settings, and releases pose-based checkpoints and a
           large Indian-SL pretraining set. [VERIFY] checkpoint availability,
           licences and code compatibility before depending on it.
    HOW  : try OpenHands/INCLUDE pretrained pose models as initial weights or
           as an ensemble member; fine-tune on our vocabulary.
    COST : integration effort; fallback = train our own (Section 5.5).

L7  PERSONAL ENROLLMENT (biggest single lever for Demo-100)
    WHAT : the demo signers record their OWN examples of each demo sign; the
           models are fine-tuned on them.
    WHY  : most of the accuracy gap in the literature is "new person, new
           camera". If the system has already seen this person, in this room,
           on this camera, that gap mostly disappears. Lab datasets with fixed
           setups and many repetitions reach ~98-99% for exactly this reason.
    HOW  : an Enroll mode guides the signer through R = 30+ repetitions per
           sign across at least 3 separate sessions; a quality gate rejects bad
           clips; a short fine-tune job runs; a report shows accuracy on the
           held-out session. Do a mini top-up (5 reps per sign) at the demo
           venue just before presenting (lighting differs).
    COST : about 30-40 minutes of recording per signer. Worth it.
    NOTE : this is a legitimate "user-adapted" system; the paper must say so
           and report signer-independent numbers separately (Tier 2).

L8  ENSEMBLE OF DIVERSE MODELS + CALIBRATED FUSION
    WHAT : run 3+ different models and combine their probabilities.
    WHY  : different models make different mistakes; averaging cancels many.
           Published work shows ensembles beat single models, and that
           combining pose-based and RGB-based models is consistently the best
           (state-of-the-art isolated-sign results use ensembles and
           pose+RGB fusion).
    HOW  : members (choose by what fits the hardware):
             M1  pose-sequence model A (TCN or BiGRU)
             M2  pose-sequence model B (small Transformer; pretrained init if
                 available)
             M3  RGB-crop video model (small 3D-CNN or video transformer,
                 pretrained backbone) -- needs a GPU for comfort; optional
             (extra seeds of M1/M2 still help if M3 is not feasible)
           Calibrate each model's probabilities on validation data
           (temperature scaling), then fuse with weights tuned on validation
           (simple grid search is enough; fancier optimisers optional).
    COST : more training and inference time; keep each model small.

L9  TEST-TIME AUGMENTATION (TTA) AND VOTING
    WHAT : classify several slightly different views of the same clip and
           combine.
    WHY  : removes sensitivity to exact start/end frames and tiny jitter.
    HOW  : e.g. 5 variants: 3 temporal resamplings (trim 0 / 3 / 6 frames at
           each end) x small scale shifts; average probabilities; also record
           how many variants agree (feeds the decision engine).
    COST : ~5x inference on a short clip; fine.

L10 DECISION ENGINE: ACCEPT / CONFIRM / REJECT
    WHAT : turn probabilities into an action, not just a label.
    WHY  : this is what converts "98% raw accuracy" into "100% accepted
           accuracy": uncertain cases are routed to a human tap.
    HOW  :
        ACCEPT  if fused top-1 prob >= T_accept
                    AND margin (top1 - top2) >= M_accept
                    AND all ensemble members agree on top-1 (or >= k of n)
                    AND TTA agreement >= A_accept
                    AND clip quality gate passed.
        CONFIRM if not accepted but top-1 >= T_confirm: show top-3 chips,
                    user taps the right one ("Did you mean ...?").
        REJECT  otherwise: "Not recognised -- please sign again".
        Choose thresholds from the risk-coverage curve on VALIDATION data:
        the loosest thresholds that still give zero accepted errors. Then
        verify on the untouched TEST session. Never tune on test.
    COST : some signs need a tap. Coverage target >= 90%.

L11 CONTEXT PRIOR ("SCRIPT / CONVERSATION MODE")
    WHAT : narrow the candidates using conversation context.
    WHY  : if the app knows the demo is a "doctor visit" scenario, then after
           "hello" only 4-5 signs are plausible next, so mistakes are rarer.
           This is how speech recognition uses language models; it is a
           legitimate technique, but must be DISCLOSED in the paper/demo.
    HOW  : config/scenarios.json lists scenarios and allowed-next-sign sets
           (or soft priors). The fused probabilities are multiplied by the
           prior and re-normalised. OFF by default; ON in scripted demos;
           always report results with it off AND on.
    COST : a little config; transparency required.

L12 REGRESSION GUARD: THE GOLDEN ACCEPTANCE TEST
    WHAT : a fixed set of recorded clips that must always pass.
    WHY  : stops "improvements" from silently breaking the demo.
    HOW  : data/golden/<signer>/<sign>/<clip> from a HELD-OUT session; the
           command `pytest -m acceptance` runs the full pipeline and asserts:
           accepted accuracy == 100%, coverage >= 90%, zero wrong-accepts.
           Run before every commit that touches models, features, thresholds
           or the decision engine.

IDEAS CONSIDERED AND REJECTED (so nobody wastes time re-evaluating)
    - Asking a general multimodal LLM (e.g. Gemini with video) to recognise the
      sign: not validated for ISL; unpredictable; slow; cannot be reproduced
      for the paper. Not on the critical path. (May be tested later purely as
      an experiment, never as the decision-maker.)
    - Continuous (sentence-level) sign translation models: need large
      sentence-aligned ISL data that we do not have.
    - Streaming sliding-window classification: shown to flap (see L3).
    - Fingerspelling alphabet: static, easy, but a different project and not
      what the INCLUDE vocabulary covers.
    - Mirroring tricks: unnecessary; handle handedness explicitly (5.3).
    - Chasing the whole 263-sign vocabulary: lowers accuracy and demo
      reliability for no gain in the headline result.


================================================================================
SECTION 4.  EXTERNAL LANDSCAPE (researched Oct 2026)
================================================================================
4.1  Dataset: INCLUDE (IIT Madras / AI4Bharat). 263 ISL word signs, 15
     categories, about 4,287-4,292 videos, one sign per video, signed by deaf
     students from a school in Chennai. Official train.csv/test.csv split plus
     INCLUDE-50. Zenodo record 4010759. [VERIFY] licence before redistributing
     clips.

4.2  Code and pretrained resources
     - AI4Bharat/INCLUDE : official code (MediaPipe Hands + BlazePose
       keypoints, transformer baseline, pretrained models). Our benchmark.
     - AI4Bharat/OpenHands : pose-based sign-recognition library with released
       checkpoints for 6 sign languages including Indian, and Indian-SL
       pretraining data. Use as pretrained starting point. [VERIFY]
     - aju22/Real-Time-ISL-Translation : MediaPipe pose + LSTM, greetings only.
     - shag527/Indian-Sign-Language-Recognition : static alphabets + a reverse
       mode (spoken word -> sign images): idea for Direction B stretch.
     - ThrisheiyanUK/Indian-Sign-Language-Recognition-System : static classes
       + speech output + data-collection tool.
     - FangyunWei/SLRT : TwoStream-SLR/SLT, spoken-to-sign avatars (heavy).
     - LucknowAI/Sign_Language_Translator : curated paper/repo list.

4.3  Evidence behind the accuracy stack
     - Ensembles of diverse models beat single models for isolated sign
       recognition (Sensors 2022, "One Model is Not Enough"; state-of-the-art
       WLASL300 result used an optimised ensemble).
     - Two-stream (RGB + pose) ensembles with Swin-Transformer backbones report
       state-of-the-art isolated-sign accuracy on WLASL/MS-ASL/ASL-Citizen
       (a 2025 ScienceDirect paper, "Ensemble transformer-based word-level sign
       language recognition with multi-modal input fusion" -- [VERIFY] venue).
     - Pose + RGB fusion is consistently best in a 2025 multi-view benchmark
       paper (WACV 2025).
     - SAM-SLR-style multi-modal ensembles: on the lab-controlled SLR500
       dataset keypoints alone ~98.2% top-1, keypoints+RGB ~99.0%
       (arXiv 2110.06161). Lab conditions + repetition => near-ceiling.
     - A study of MediaPipe vs OpenPose vs MMPose found MediaPipe hand
       keypoints better, and depth values hurt (arXiv 2306.17558).
     - Pretraining helps in low-resource sign recognition (OpenHands, ACL 2022).

4.4  MediaPipe facts
     - Holistic = 33 pose + 21 per hand + 468 face landmarks.
     - Legacy mp.solutions.* APIs are unsupported; use MediaPipe Tasks (Python
       and Web). [VERIFY] class names and versions at install time.
     - If a hand is not detected, no values return for that hand: handle
       missing hands explicitly.

4.5  Facts about the reference codebase that drive decisions
     - Gemini 2.0 Flash is retired (Google lists gemini-2.0-flash and -flash-lite
       as shut down; named replacements gemini-3.5-flash / gemini-3.1-flash-lite).
       One source also lists gemini-2.5-flash for retirement on 16 Oct 2026.
       => model name must be an env var. [VERIFY]
     - The old code emitted a "sentence" after EVERY word (no sentence buffer).
     - Docs said 3 agreeing predictions + 1.5 s cooldown; one service used 2 and
       1.0 s (docs/code drift).
     - The mobile path skipped smoothing/velocity features the server used
       (train/serve skew).
     - Reported 89.4% was on a custom stratified split, not comparable to the
       INCLUDE paper; dataset count differed (4,257 vs ~4,292).


================================================================================
SECTION 5.  TARGET ARCHITECTURE (v3)
================================================================================
5.0  Big idea in plain words
     The browser is a "smart camera": it checks the setup, decides when a sign
     starts and ends, and uploads the whole clip. The server is the "expert":
     it extracts hand/body detail from every frame, runs several models,
     combines them carefully, and only announces a word when it is sure --
     otherwise it asks the user or says "please sign again". Words are then
     assembled into a sentence, translated and spoken.

5.1  Diagram

 BROWSER (React web app)                        SERVER (FastAPI, Python 3.10/3.11)
 -----------------------                        ----------------------------------
 Webcam 720p/1080p @30fps
   |                                             POST /api/recognize  (whole clip)
 PRE-FLIGHT CHECK (light MediaPipe Web)             |
   hands visible? light? framing?                   1. decode every frame
   |                                                2. LANDMARK EXTRACTOR (MediaPipe
 CAPTURE CONTROLLER                                    Tasks Holistic, high accuracy)
   auto start/stop on hand raise/rest               3. QUALITY GATE (hands visible
   OR hold Space (push-to-sign)                        fraction, motion, duration)
   |                                                4. FEATURE BUILDER (shared lib)
 clip buffer (frames or WebM)                       5. MODELS: M1 pose-TCN/GRU
   | upload ---------------------------------->        M2 pose-Transformer
                                                         M3 RGB-crop video (optional)
 UI shows "Analysing..."                            6. TTA (several time/scale views)
   |                                                7. CALIBRATE + FUSE (+ optional
 RESULT  <-----------------------------------         context prior, L11)
   ACCEPT  -> word chip appears                     8. DECISION ENGINE
   CONFIRM -> top-3 chips, user taps                   accept | confirm | reject
   REJECT  -> "please sign again"                   9. log everything (clip id,
   |                                                   probs, decision) for audit
 SENTENCE BOX (editable)
   words collect; pause 2.5 s or "Finish"        POST /api/compose  (sentence layer)
   |  ----------------------------------------->   scripted sentence table
   |                                                -> else LLM (JSON en/hi/kn)
 SUBTITLES + SPEECH (browser voices)                -> else template fallback
   |
 MIC -> /ws/speech -> VAD + Whisper -> live captions (Direction B)
 ENROLL MODE -> /api/enroll/* -> quality gate -> store -> fine-tune job -> report

5.2  Walk-through of one sign (what happens, step by step)
     1. Signer stands in frame; pre-flight lights go green.
     2. Signer raises hands; the capture controller starts buffering frames.
     3. Signer finishes and lowers hands; controller stops (or releases Space).
        Whole clip (~1-3 s, ~30-90 frames) is uploaded.
     4. Server extracts landmarks from every frame and checks quality (e.g.
        hands visible in >= 70% of frames; some motion; 0.4-5 s long). If it
        fails -> REJECT with a reason ("keep both hands in view").
     5. Features are built and each model produces probabilities for the
        vocabulary; TTA repeats this for several views; results are calibrated
        and fused.
     6. Decision engine: ACCEPT (confident and everyone agrees), CONFIRM
        (probably one of three), or REJECT.
     7. UI shows the outcome; accepted/confirmed words enter the sentence box.
     8. After a pause or "Finish", the sentence layer builds, translates, shows
        and speaks the sentence (the user can edit before speaking in strict
        mode).

5.3  Capture and landmark details
     - Browser pre-flight uses light MediaPipe Web only to give live feedback;
       it is NOT the accuracy-critical path. The server re-extracts landmarks
       from the uploaded frames with the best settings.
     - Capture 30 FPS; upload JPEG frames at quality ~85 (or a WebM). Do not
       downscale below 640 px wide; prefer 720p.
     - Do NOT mirror the video that is uploaded. The on-screen preview MAY be
       mirrored with CSS.
     - Handedness: do not trust the "Left/Right" labels blindly (they flip with
       mirroring). Decide in Phase 1 (order hands by position relative to the
       body midline, or test both) and record it in docs/DECISIONS.md.
     - Landmark set per frame: pose subset (nose, shoulders, elbows, wrists,
       hips = MediaPipe pose indices 0, 11, 12, 13, 14, 15, 16, 23, 24), 21
       landmarks per hand; optional few face points. 2D (x, y) only.
     - Missing hand => zeros + presence flag; short gaps (<= 3 frames) are
       interpolated; longer gaps remain zero + flag.

5.4  Feature builder (module signtalk_core.features) -- the SINGLE source of
     truth for features; training and inference both import it.
     - Origin = shoulder midpoint; scale = shoulder width.
     - Channels per frame: body-relative positions (pose subset + both hands),
       hand-local coordinates (relative to each hand's wrist, scaled by palm
       size), frame-to-frame velocities, hand-presence masks. Exact size is
       computed in code (roughly 200 dims).
     - Smoothing: small moving average (window 3), applied before resampling.
     - Resample the clip to T frames (default 32, linear interpolation) for the
       sequence models. The RGB branch samples T_rgb frames (default 16) and
       crops using landmarks.
     - Golden-vector tests: saved input -> saved expected output; fail on any
       change so training/inference can never silently diverge.

5.5  Models (all PyTorch; export ONNX for serving)
     - M1: temporal conv network or 2-layer BiGRU on [T, F].
     - M2: small Transformer encoder (about 4 layers, d_model 128, 4 heads).
       Initialise from a pretrained pose checkpoint if one is usable [VERIFY].
     - M3 (optional, GPU): RGB-crop video classifier with a pretrained video
       backbone [VERIFY which one fits the hardware]; fed with hand/face/upper
       body crops.
     - Training: label smoothing 0.1, dropout 0.3, weight decay, early stopping
       on validation, fixed seeds, several seeds per architecture.
     - Augmentation (landmark space): small rotation (+-10 deg), scale (+-10%),
       translation, jitter, temporal speed (0.8-1.2x), random frame drop,
       left-right hand-swap (flip x AND swap hand channels; only after the
       handedness decision is verified). For RGB: colour jitter, small crops.
     - Each model is exported to ONNX and parity-tested against PyTorch.

5.6  Calibration and fusion (module signtalk_core.fusion)
     - Fit a temperature per model on validation data so confidences mean what
       they say.
     - Fuse by weighted average of calibrated probabilities; tune weights on
       validation (grid search). Keep the weights in config.
     - Output: fused probabilities, per-model top-1, agreement count, TTA
       agreement, margin.

5.7  Decision engine (module signtalk_core.decision) -- see L10. Thresholds
     live in config; a script `tools/pick_thresholds.py` computes them from the
     validation risk-coverage curve and writes a report.

5.8  Enrollment (Phase 4)
     - UI: choose signer profile -> for each demo sign, the app shows a
       reference picture/clip and records repetitions with a counter; quality
       gate rejects bad ones; progress bars per sign.
     - Rules: at least 3 sessions, at least 30 clips per sign per signer, vary
       distance/lighting slightly between sessions (do not overdo it).
     - Server stores clips and landmarks under data/enroll/<signer>/<session>/.
     - A fine-tune job (script + button) fine-tunes models on INCLUDE + enrolled
       data, evaluates on the held-out session, writes a report, and only
       promotes the model to "active" if the report meets the targets.
     - Venue top-up: 5 reps per sign on the day, quick fine-tune.

5.9  Context prior / script mode (L11) -- config/scenarios.json:
       {"scenario":"doctor_visit","start":["hello","help"],
        "next":{"hello":["help","pain","water"], ...}}
     Used only when enabled; always logged so results can be reported both ways.

5.10 Sentence layer
     - SENTENCE BUFFER: words collect until (a) no new word for ~2.5 s, (b)
       user taps Finish, or (c) 8 words.
     - ORDER OF PREFERENCE (determinism first):
         1. SCRIPTED TABLE: config/sentences.json maps known word sequences to
            pre-verified sentences in en/hi/kn (verified by a native speaker).
            No LLM, no hallucination risk. Cover every sentence in the demo.
         2. LLM: one call returning strict JSON {"en","hi","kn","tone"}. Rules
            in the prompt: use only the signed meaning, add nothing, keep it
            short. Validate the JSON schema; retry once.
         3. TEMPLATE FALLBACK: capitalise/punctuate; join words; phrasebook for
            translation.
     - SAFETY CHECK on LLM output: every signed word (or a synonym from a small
       list) must appear in the English sentence; otherwise discard and fall
       back to template.
     - "Strict mode" (default for demo): the sentence appears editable and is
       spoken only after the user taps Confirm. "Fast mode": speak
       automatically.
     - Model name from env var GEMINI_MODEL; timeout 4 s; provider interface so
       another LLM (or a local Ollama model) can be swapped in. [VERIFY] names.

5.11 Translation
     - Scripted table and the LLM's JSON already provide en/hi/kn. Cache by
       hash(text+lang) in SQLite. Phrasebook JSON for offline. Google Cloud
       Translate is optional (needs billing). Hindi and Kannada text MUST be
       reviewed by a native speaker before the demo.

5.12 Speech
     - Text-to-speech: browser SpeechSynthesis by default (free, works offline;
       Hindi/Kannada voice availability varies by device -- test on the demo
       machine). Optional gTTS via /api/tts when online. Drop Coqui TTS.
     - Speech-to-text: faster-whisper or openai-whisper (small/base) over
       /ws/speech with voice-activity detection; Kannada is generally weaker
       than English/Hindi, so test 20 sentences early [VERIFY]. Browser Web
       Speech API as a fallback.

5.13 Emotion (optional, low priority)
     - Default "neutral". If enabled, run DeepFace on one face crop per second
       on the server with a rolling vote; it only nudges sentence tone. Never
       blocks recognition. Because video already goes to the server, no extra
       privacy cost.

5.14 Backend structure
     - Thin FastAPI routers; ALL ML logic in the installable package
       `signtalk_core`. Pydantic validation on every request/message; size
       limits; rate limiting; CORS with explicit origins; structured logs with a
       request id; /metrics with per-stage latency histograms.
     - Auth optional: "guest" mode by default; Firebase can be enabled by
       config. Never put a JWT in a WebSocket URL; use a short-lived ticket.
     - Storage: SQLite (conversations, enrollment index, decisions log),
       Firestore optional behind a Storage interface.
     - Every recognition decision (clip id, per-model probs, fusion, decision,
       thresholds used) is logged: this is the audit trail for the paper.

5.15 Offline mode (simplified vs v2)
     Because the server runs on the demo laptop, "offline" simply means the
     laptop has no internet: recognition, Whisper, scripted sentences, phrasebook
     and browser TTS all work locally; only the LLM falls back to scripted/
     template sentences. No browser-side ML port is needed. Acceptance: with
     Wi-Fi off, a rehearsed sentence is recognised, shown and spoken.

5.16 Latency budget (measure; targets, not promises)
     clip duration 1-3 s (the sign itself) + upload < 0.5 s + landmark
     extraction + models + TTA + fusion: aim < 3 s on a GPU laptop, < 6 s CPU
     only. Report each stage separately. Never claim "300 ms end-to-end".


================================================================================
SECTION 6.  INTERFACE CONTRACTS v3 (change only with human approval)
================================================================================
6.1  POST /api/recognize   (multipart or JSON)
     request : clip (WebM/MP4) OR frames[] (base64 JPEG), fps, signer_id,
               scenario_id|null, client_quality{hands_visible_pct,brightness}
     response: {
       "clip_id": str,
       "decision": "accept"|"confirm"|"reject",
       "label": str|null,                 # set when accept
       "candidates": [{"label":str,"p":float}, ... up to 3],
       "confidence": float, "margin": float,
       "agreement": {"models": "3/3", "tta": "5/5"},
       "reject_reason": null|"low_quality"|"hands_not_visible"|"too_short"|
                        "too_long"|"low_confidence",
       "latency_ms": {"extract":int,"models":int,"fusion":int,"total":int}
     }
6.2  POST /api/confirm   {clip_id, chosen_label}   (user tap on a candidate;
     stored as a labelled sample for review)
6.3  POST /api/compose
     request : {"words":[str],"emotion":str,"history":[str],"scenario_id":str|null}
     response: {"sentences":{"en":str,"hi":str,"kn":str},
                "source":"scripted"|"llm"|"template","verified":bool,
                "latency_ms":int}
6.4  WS /ws/speech  server -> client {"type":"transcript","text":str,
                                      "is_final":bool,"lang":str}
6.5  Enrollment: POST /api/enroll/start {signer_id}, POST /api/enroll/clip
     {signer_id,session_id,sign,clip}, GET /api/enroll/progress/{signer_id},
     POST /api/enroll/train {signer_id}, GET /api/enroll/report/{signer_id}
6.6  GET /api/vocab -> [{"label","category","demo":bool}]
6.7  GET /health, GET /metrics
6.8  Library contract (Python):
        recognize_clip(frames, fps, signer_id, scenario_id=None) -> dict
        shaped like the 6.1 response. T, F, thresholds, weights come from config.
6.9  Config contract: ONE file config/signtalk.yaml holds every tunable.


================================================================================
SECTION 7.  REPOSITORY LAYOUT
================================================================================
signtalk/
  README.md
  docs/ VISION.md (this file) PROGRESS.md DECISIONS.md EVALUATION.md
  config/ signtalk.yaml  vocabulary.json  scenarios.json
          sentences.json (scripted en/hi/kn)  phrasebook.json
  core/signtalk_core/
      landmarks.py features.py quality.py augment.py
      models/ (pose_tcn.py pose_transformer.py rgb_video.py)
      calibration.py fusion.py decision.py tta.py recognize.py
  core/tests/  (unit + golden-vector + acceptance)
  training/
      prepare_include.py   train.py   evaluate.py   export_onnx.py
      finetune_enrolled.py   collect_clips.py   notebooks/
  tools/ pick_thresholds.py  pick_vocabulary.py  make_report.py
  server/app/ (routers, schemas, providers, storage, jobs)  server/tests/
  web/src/ (React + Vite + Tailwind; preflight, capture, enroll, sentence UI)
  data/   (gitignored) raw/ processed/ enroll/ golden/ feedback/
  models/ (gitignored or LFS) *.onnx labels.json model_card.md
  scripts/ setup, run_all, lint
  .env.example
Conventions: Python 3.10/3.11, type hints, ruff+black, pytest (markers: unit,
acceptance), pin dependency versions after a clean install, commit lockfiles.
Training compute: use a free Colab/Kaggle GPU if the laptop has none.


================================================================================
SECTION 8.  DATA PLAN
================================================================================
8.1  INCLUDE: download the categories needed (INCLUDE-50 + candidate demo
     signs). Keep official split files. [VERIFY] licence.

8.2  Preprocess (training/prepare_include.py): video -> all frames ->
     landmarks (same extractor as the server) -> quality gate -> features ->
     .npz; plus the cropped RGB frames if M3 is used. Log every failed/skipped
     clip with a reason (fixes the old 4,257-vs-4,292 mystery). Per-class
     hand-detection rate report.

8.3  Vocabulary selection (tools/pick_vocabulary.py): candidates -> quick
     model -> confusion analysis -> drop confusable pairs -> final 10-20 in
     config/vocabulary.json. Also keep an UNKNOWN set (other INCLUDE signs,
     idle motion, random gestures) used as negatives for rejection testing.

8.4  Enrollment sessions: see 5.8. Directory: data/enroll/<signer>/<session>/.
     Metadata per clip: signer, session id, date/time, lighting note, camera,
     distance note.

8.5  SPLITS (to prevent leakage, Section 1.6)
     - Official INCLUDE split for all INCLUDE-only numbers.
     - Enrolled data: sessions 1-2 train, session 3 validation (thresholds),
       a LATER session 4 = final test, touched once. Rotate for cross-checks.
     - Signer-independent number: hold out one whole team member entirely.
     - GOLDEN set = clips from a held-out session, frozen in data/golden/.

8.6  Active learning: confirm taps and user corrections go to
     data/feedback/pending/; a human reviews before they join training. Never
     auto-train on unreviewed data.


================================================================================
SECTION 9.  EVALUATION PLAN (feeds the paper)
================================================================================
9.1  Offline INCLUDE: top-1/top-5, macro-F1, per-class table, confusion
     matrix on the official split (INCLUDE-50, and chosen vocabulary).
9.2  Demo-100: raw, accepted, coverage, final accuracy (metrics A-D), with the
     risk-coverage curve, on the held-out enrolled session. State n (number of
     trials) and the rule-of-three bound (1.4).
9.3  Signer-independent: hold-out signer results (Tier 2).
9.4  Ablations (each lever measured alone and cumulatively):
        pose only vs pose+hands; +velocity; +hand-local; +augmentation;
        single model vs ensemble; +RGB branch; +TTA; +enrollment;
        whole-clip vs sliding window (shows the flapping fix);
        context prior on vs off.
9.5  Open-set: wrong-accept rate on UNKNOWN set (target: 0 accepted wrong
     words; report n and the bound).
9.6  Live trials: 3+ signers x 2 lighting conditions x 10 reps x every demo
     sign; raw trial sheets saved. Aim for n >= 300 trials for the headline
     (gives a ~1% upper bound if zero errors).
9.7  Latency: per-stage p50/p95 on the demo machine.
9.8  System: kill the LLM / network / Firebase and confirm the app still
     recognises, shows and speaks the scripted sentences.
9.9  Report limitations plainly (small vocabulary, few signers, enrolled-user
     setting, domain gap, one school's students in INCLUDE).


================================================================================
SECTION 10.  BUILD PHASES AND GATES  (full plan ~5 weeks; minimum path below)
================================================================================
PHASE 0 -- Setup and baseline (2-3 days)
  Repo skeleton, clean environments, download INCLUDE-50 categories + candidate
  signs, load official split, run any simple baseline (or the official INCLUDE
  code) to get a measured number.
  GATE 0: baseline number on INCLUDE-50 official test in docs/PROGRESS.md.

PHASE 1 -- WALKING SKELETON (3-4 days)
  Browser records a clip (Space-hold) -> /api/recognize -> MediaPipe landmarks
  -> one simple pose model on 5 signs -> word shown on screen. Ugly is fine;
  end-to-end is the point.
  GATE 1: a team member signs 5 words and sees them (accuracy irrelevant);
  handedness decision written; golden-vector tests exist for features.

PHASE 2 -- Models and ensemble (5-6 days)
  Full preprocessing with failure report; M1 and M2 (and M3 if GPU); pretrained
  initialisation attempt; calibration; fusion; TTA; vocabulary selection;
  export ONNX with parity tests.
  GATE 2: official-split INCLUDE-50 numbers recorded for each model and the
  ensemble; ablation table started; ONNX parity passes.

PHASE 3 -- Decision engine and capture polish (3-4 days)
  Pre-flight check, auto start/stop, quality gate, decision engine with
  thresholds from validation, accept/confirm/reject UI, regression guard
  scaffolding.
  GATE 3: accepted-accuracy and coverage measured on a held-out recording set;
  results in PROGRESS.md.

PHASE 4 -- Enrollment and Demo-100 (5-6 days)
  Enrollment UI + storage + fine-tune job + report; record sessions 1-4 for all
  demo signers; fine-tune; thresholds; golden set frozen.
  GATE 4: `pytest -m acceptance` passes (accepted accuracy 100%, coverage >=
  90%) on the held-out session; n and bound stated.

PHASE 5 -- Language layer (3-4 days)
  Sentence buffer; scripted sentence table (native-speaker verified); LLM
  provider with safety check and fallbacks; language switch; browser TTS;
  /ws/speech with Whisper + VAD; captions + history.
  GATE 5: with no LLM key, every demo sentence still appears and speaks in all
  three languages; with the key, p95 compose latency recorded.

PHASE 6 -- Evaluation and paper assets (3-4 days)
  Run Section 9 completely; produce tables/figures; fix claims (Section 14).
  GATE 6: docs/EVALUATION.md complete; every number traceable to a logged run.

PHASE 7 -- Demo hardening (2-3 days)
  Venue top-up procedure, rehearsal script, backup video, one-command startup,
  clear error messages, lighting guide on screen.
  GATE 7: three clean dry runs by two different people, with Wi-Fi off once.

MINIMUM PATH IF ONLY ~10 DAYS REMAIN (do exactly this, in this order)
  1. Phase 1 skeleton (days 1-2).
  2. Choose 8-10 distinct signs; record enrollment (3 sessions x 30 reps) for
     the demo signers (day 3-4).
  3. Train M1 + M2 (+ extra seeds), calibrate, fuse, TTA (days 4-6).
  4. Decision engine + confirm UI + thresholds (day 6-7).
  5. Scripted sentences + browser TTS + Hindi/Kannada verification (day 7-8).
  6. Golden acceptance test + dry runs (day 9-10).
  Skip: RGB model, pretrained integration, speech Direction B polish, emotion.

STRETCH (only after Gate 7): RGB branch if skipped; sign pictures for
Direction B; grow vocabulary toward 50-100 (accept lower accuracy and report
it); emotion tuning.


================================================================================
SECTION 11.  RISKS AND FALLBACKS
================================================================================
R1  Live accuracy below offline (domain gap)  -> enrollment (L7), venue top-up,
    pre-flight check, constrained vocabulary.
R2  Hands not detected (light, sleeves, speed) -> quality gate + clear message,
    plain dark sleeves, front light, 30 FPS, higher input resolution.
R3  Auto start/stop misfires -> hold-Space push-to-sign always available; show
    capture state on screen.
R4  Ensemble too slow on laptop -> drop M3, reduce TTA to 3 views, run models
    in ONNX; measure first.
R5  LLM unavailable / model retired / quota -> scripted sentence table first,
    env-var model name, 4 s timeout, template fallback.
R6  Hindi/Kannada errors -> native-speaker verification of scripted and
    phrasebook text; flag LLM output as unverified in the UI.
R7  Whisper weak in Kannada -> test early; smaller model; browser STT fallback;
    scope Kannada STT as best-effort.
R8  Overfitting to the demo / inflated numbers -> session-based splits (1.6),
    untouched final test, golden set, honest tiers in the paper.
R9  Threshold overfitting -> tune on validation session only; final test once.
R10 Time overrun -> follow the minimum path; phases are ordered by value.
R11 Dependency conflicts -> single ML framework (PyTorch), pinned lockfiles, no
    Coqui TTS, ONNX for serving.
R12 Dataset licence / consent -> do not redistribute INCLUDE clips; consent
    notice for team/friend recordings; data folder gitignored.
R13 Venue differences (light, camera) -> venue top-up (5 reps per sign),
    on-screen lighting guide, bring own webcam and a light.


================================================================================
SECTION 12.  ENGINEERING RULES FOR THE AGENT
================================================================================
12.1  Small commits, one purpose each; main always runs.
12.2  Every module in core/ has unit tests; features, fusion and decision have
      golden-vector tests; models have ONNX parity tests.
12.3  Training and serving import the SAME feature code. Never re-implement.
12.4  No magic numbers: tunables live in config/signtalk.yaml.
12.5  No secrets in git; keys only in .env; maintain .env.example.
12.6  Type hints and docstrings with tensor shapes ("x: float32 [T, F]").
12.7  Fail loudly in development; degrade gracefully on the demo path: every
      external call has a timeout, a retry cap and a fallback.
12.8  Log decisions (Section 5.14); never delete logs that back a paper number.
12.9  After every gate, write the exact command and real output in
      docs/PROGRESS.md; record non-obvious choices in docs/DECISIONS.md.
12.10 Never tune on test data. Never report a number from a run you cannot
      reproduce (record seed, config, git commit hash with every run).
12.11 Run `pytest -m acceptance` before committing changes to anything that
      affects recognition.
12.12 If this file looks wrong or an external fact has changed, write it in
      DECISIONS.md and ask the human; do not silently deviate.
12.13 Accessibility: high-contrast large subtitles, keyboard-operable controls,
      no flashing.


================================================================================
SECTION 13.  CONFIGURATION AND ENVIRONMENT
================================================================================
.env.example
  GEMINI_API_KEY=
  GEMINI_MODEL=gemini-3.5-flash       # [VERIFY]; Google retires models often
  LLM_PROVIDER=gemini                 # gemini | none | ollama
  LLM_TIMEOUT_SECONDS=4
  WHISPER_MODEL=small                 # small | base | tiny
  ENABLE_EMOTION=false
  STORAGE_BACKEND=sqlite              # sqlite | firestore
  AUTH_MODE=guest                     # guest | firebase
  FIREBASE_SERVICE_ACCOUNT_PATH=
  CORS_ORIGINS=http://localhost:3000
  USE_GPU=auto

config/signtalk.yaml (starting values; tune from data, never hardcode)
  sequence_length: 32          # pose models
  rgb_frames: 16
  smoothing_window: 3
  quality: {min_hands_visible_pct: 0.7, min_seconds: 0.4, max_seconds: 5.0}
  tta: {views: 5, trim_frames: [0, 3, 6]}
  decision:
    t_accept: 0.85  m_accept: 0.30  min_model_agreement: all  tta_agreement: 0.8
    t_confirm: 0.40
  sentence: {idle_seconds: 2.5, max_words: 8, mode: strict}
  context_prior: {enabled: false, scenario: null}
(The numbers above are placeholders; tools/pick_thresholds.py replaces the
 decision thresholds with values computed from validation data.)


================================================================================
SECTION 14.  PAPER NOTES (claims to get right)
================================================================================
- Call it isolated sign recognition, not continuous translation.
- State the landmark source accurately (MediaPipe hands + upper-body pose here;
  fix any "17" or "33" keypoint wording so it matches the final system).
- Headline in three tiers (1.5): Demo-100 (closed set, enrolled signers),
  signer-independent 50-sign, and INCLUDE official-split numbers. Cite the
  INCLUDE paper's figures as published results, not ours.
- Report metrics A-D, the risk-coverage curve, n, and the rule-of-three bound.
  Never write "100% accuracy" without "on N trials, closed vocabulary of K
  signs, enrolled signers, abstain/confirm enabled".
- Disclose enrollment (L7), the decision engine (L10) and context prior (L11)
  explicitly; show results with and without each.
- Latency: per-stage numbers; separate "processing time" from the time a sign
  takes to perform.
- Ablation table (9.4) and whole-clip vs sliding-window comparison.
- Limitations (9.9) and an ethics paragraph: community involvement, deaf
  signer feedback where possible, no claim of replacing interpreters, consent
  for recordings.
- Related work: INCLUDE, OpenHands, iSign, SAM-SLR, ensemble SLR papers,
  Uni-Sign, SignLLM, TwoStream-SLT, MediaPipe holistic, the open-source ISL
  repos in 4.2.


================================================================================
SECTION 15.  DEMO PLAN
================================================================================
15.1  Setup (before the audience arrives): front light, plain background, solid
      dark long sleeves, camera at chest height, ~1 m distance, run the
      pre-flight check, do the 5-reps-per-sign venue top-up, run `pytest -m
      acceptance` on the latest golden clips plus 3 fresh live passes.
15.2  Script (~6 minutes)
      1. Explain the idea in one sentence; show the pre-flight lights.
      2. Sign 3 words one at a time; show word chips and the confidence.
      3. Sign a 3-word scripted sentence; show the sentence in English, then
         Hindi and Kannada; play speech.
      4. Show "please sign again" with a made-up gesture (the system refuses to
         guess) -- turn this into a feature, not a failure.
      5. Show a "confirm" case (pick a borderline sign) to explain
         abstain/confirm.
      6. Hearing person speaks; live captions.
      7. Switch Wi-Fi off; repeat step 3.
      8. Show the metrics page: risk-coverage curve, per-stage latency, the
         ablation table.
15.3  Backups: pre-recorded screen capture, second laptop with the same build,
      a spare webcam and light, cached scripted sentences.


================================================================================
SECTION 16.  GLOSSARY (for the team)
================================================================================
Landmarks      : points the computer finds on a body/hand (e.g. fingertip,
                 wrist) in each video frame.
Sequence model : a neural network that reads a series of frames over time.
Ensemble       : several different models whose answers are combined.
Calibration    : adjusting a model's confidence so "90% sure" is right about
                 90% of the time.
TTA            : test-time augmentation; classify several slightly altered
                 versions of the same clip and combine.
Margin         : difference between the best and second-best probability.
Abstain        : the system declines to answer rather than guess.
Risk-coverage  : curve showing how accuracy changes as the system answers
                 more or fewer cases.
Enrollment     : recording a person's own examples so the system adapts to them.
Data leakage   : test data that is too similar to training data, inflating
                 accuracy.
Signer-independent : tested on people the model never saw in training.
Golden set     : frozen clips that must always be recognised correctly.
ONNX           : a portable model format for fast inference.
VAD            : voice activity detection (finds when someone is speaking).
Gloss          : the word label for a sign.


================================================================================
SECTION 17.  REFERENCES
================================================================================
INCLUDE dataset        https://zenodo.org/records/4010759
INCLUDE code           https://github.com/AI4Bharat/INCLUDE
OpenHands (AI4Bharat)  https://github.com/AI4Bharat/OpenHands
OpenHands paper        https://arxiv.org/abs/2110.05877
SAM-SLR multi-model    https://arxiv.org/abs/2110.06161
Ensembles for ISLR     https://pmc.ncbi.nlm.nih.gov/articles/PMC9269724/
Sign embeddings study  https://arxiv.org/abs/2306.17558
iSign                  https://arxiv.org/abs/2407.05404
Uni-Sign               https://arxiv.org/abs/2501.15187
TwoStream-SLR/SLT      https://arxiv.org/abs/2211.01367
SLRT repo              https://github.com/FangyunWei/SLRT
Real-Time ISL          https://github.com/aju22/Real-Time-ISL-Translation
ISL alphabets/reverse  https://github.com/shag527/Indian-Sign-Language-Recognition
ISL system + speech    https://github.com/ThrisheiyanUK/Indian-Sign-Language-Recognition-System
Paper/repo list        https://github.com/LucknowAI/Sign_Language_Translator
MediaPipe solutions    https://developers.google.com/edge/mediapipe/solutions/guide
Gemini deprecations    https://ai.google.dev/gemini-api/docs/deprecations


================================================================================
SECTION 18.  OPEN QUESTIONS FOR THE HUMAN TEAM (answer during Phase 0)
================================================================================
Q1. Demo laptop: CPU, GPU (model, VRAM), RAM, OS? (decides M3 and Whisper size)
Q2. How many days remain until the demo and until paper submission?
    (decides full plan vs minimum path)
Q3. Who are the demo signers (how many people), and will they be available for
    ~40 minutes each for enrollment across 3+ sessions?
Q4. Is there access to someone fluent in ISL (a signer or teacher) to check
    that our demo signs are performed correctly?
Q5. Is a Gemini (or other LLM) API key available, with billing if needed?
Q6. Who can verify Hindi and Kannada sentences?
Q7. Which 10-20 signs matter most for the demo story (a doctor visit? a shop?
    school?) -- this picks the vocabulary and scenario.
Q8. Is Firebase required by the department, or is local SQLite acceptable?
Q9. Is a mobile app required, or is a responsive web app enough?
Q10. Is the demo room's lighting and camera known/controllable in advance?

================================================================================
END OF DOCUMENT
================================================================================

SUGGESTED FIRST PROMPT TO THE AGENT
-----------------------------------
"Read docs/VISION.md completely. Do PHASE 0 only: create the repo skeleton from
Section 7, set up clean Python and Node environments, download the INCLUDE-50
categories, load the official split, and get any simple baseline's measured
INCLUDE-50 official-test number. Record the exact commands and real outputs in
docs/PROGRESS.md. Then list what you need from me using Section 18, and stop at
Gate 0."