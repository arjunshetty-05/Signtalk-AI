# Vision & Scope

Source: `PROJECT_CONTEXT.md` Sections 1–2. This file condenses the vision and the
honesty rules; read the context document for the full reasoning.

## The problem

Over 466 million people live with disabling hearing loss. Most hearing people
cannot sign, and typing back and forth is slow and awkward.

## Vision (one sentence)

A web app where a signer faces a webcam and the system **speaks and subtitles**
what they signed (English / Hindi / Kannada) with very high reliability, while
the hearing person's speech appears as live captions.

## Two directions

- **Direction A — Sign → Voice (primary):** webcam → record a sign clip →
  server recognises the sign (via the accuracy stack) → words are collected →
  a sentence is built → translated → subtitle + speech.
- **Direction B — Voice → Subtitles (secondary):** microphone →
  speech-to-text → live captions + conversation log. Stretch: show a sign
  picture/video for recognised vocabulary words.

## Honest scope statement

> "SignTalk AI performs **isolated** Indian Sign Language word recognition on a
> curated vocabulary. Recognised words are accumulated and a language model
> reconstructs a grammatical sentence. It is **not** a continuous sign-language
> translation system."

## Non-goals

- Continuous ISL (sentence-level) translation.
- Fingerspelling (unless everything else is done).
- A native mobile app (a responsive web app is enough).
- All 263 INCLUDE signs.
- Special hardware (gloves, depth cameras).

## Privacy decision (recorded so it is not re-debated)

The owner explicitly does **not** need video privacy for this college project.
Video is sent to the project's own server and may be stored for training and
evaluation. Minimum requirements: a one-line consent notice in the app, do not
publish identifiable video without consent, and keep the data folder out of git.

---

## The truth about "100% accuracy"

This is the most important section of the whole project. **No published system
recognises open-vocabulary sign language at 100%.** The INCLUDE paper (the
dataset used here) reports 94.5% on a 50-sign subset and 85.6% on all 263 signs.
Promising "100% on any sign, any person, any camera" would be false.

### What is engineerable: "DEMO-100"

A closed set of 10–20 signs, performed by **enrolled** signers (the team) in a
controlled setup, where the system is designed so that **every word shown or
spoken is correct** — achieved by stacking accuracy techniques and letting the
system **abstain or ask** when unsure rather than guess.

### Always report four metrics (never only the flattering one)

| Metric | Meaning | Target |
|---|---|---|
| **A. Raw accuracy** | Fraction of attempts where top-1 is correct (what most papers report). | — |
| **B. Accepted accuracy** | Among auto-accepted attempts, fraction correct. | **100%** on demo sessions |
| **C. Coverage** | Fraction of attempts accepted automatically. | **≥ 90%** |
| **D. Final-output accuracy** | After the user taps to confirm. | correct by construction |

You can push B to 100% simply by being cautious, but then C falls. Showing B and
C together (a **risk-coverage curve**) is the honest presentation.

### You cannot prove 100% with a finite test — the "rule of three"

Zero errors in `n` independent trials gives a 95% upper bound on the true error
rate of roughly `3/n`:

| n | 95% error upper bound |
|---|---|
| 30 | ~10% |
| 100 | ~3% |
| 300 | ~1% |
| 1000 | ~0.3% |

So "0 errors in 300 trials" supports "accuracy ≥ 99% with 95% confidence" — not
"100%". A 20-trial test proves almost nothing.

### Three accuracy tiers (expectations, not promises)

- **Tier 1 — DEMO-100:** 10–20 signs, enrolled signers, controlled setup,
  abstain/confirm on. Target accepted accuracy near 100%. This is the headline.
- **Tier 2 — 50 signs, unseen signers:** the honest generalisation number;
  expect high-80s to mid-90s (INCLUDE paper's 94.5% is the reference).
- **Tier 3 — all 263 signs:** reference-level (~85% in the literature). Not a
  goal; report only if time permits.

### The #1 accidental way to fake accuracy — DATA LEAKAGE

If test clips come from the **same recording session** as training clips,
accuracy looks great and means nothing. Rule: **test data must come from a
different session** (different day/time/lighting/position) than training data.

### The honesty rule

The demo and paper must never overstate. "100%" is only ever claimed for a
precisely defined closed demo scenario with measured trials, stated as:
"100% accuracy **on N trials, closed vocabulary of K signs, enrolled signers,
abstain/confirm enabled**." Anything outside that is reported with its real
number.
