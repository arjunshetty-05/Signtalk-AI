import { useCallback, useEffect, useRef, useState } from "react";
import { useWebcam } from "../hooks/useWebcam.js";
import { recognizeClip, confirmChoice } from "../api.js";
import VideoPreview from "../components/VideoPreview.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import PreflightCheck from "../components/PreflightCheck.jsx";
import SentenceBox from "../components/SentenceBox.jsx";
import GlassButton from "../components/GlassButton.jsx";
import Celebrate from "../components/Celebrate.jsx";

// Capture frame rate for the base64 JPEG batch sent to /api/recognize. The
// server resamples to config `sequence_length`, so this only needs to be a
// sensible real-time rate, not an exact match.
const CAPTURE_FPS = 15;

/**
 * Tap-to-sign capture screen (record-then-recognise, Section 6.1).
 *
 * TAP (don't hold) to start recording a whole clip, then TAP again to stop and
 * upload the UN-mirrored frame batch to /api/recognize — a toggle, not a hold,
 * because you can't comfortably hold a key/button while signing with both
 * hands. Space, the big button, and touch all toggle. The capture state (idle /
 * recording / analysing) is always visible, with a live elapsed timer while
 * recording.
 *
 * @param {{signerId: string, onWord?: (label: string) => void}} props
 *   `onWord` is called with each accepted/confirmed word so the parent can
 *   collect them into the sentence buffer.
 */
export default function CaptureScreen({ signerId, onWord }) {
  const { videoRef, ready, error, startCapture, stopCapture } = useWebcam();
  const [state, setState] = useState("idle"); // idle | recording | analysing
  const [frameCount, setFrameCount] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [result, setResult] = useState(null); // recognize response
  const [message, setMessage] = useState("");
  const [confirmed, setConfirmed] = useState(null); // label the user tapped
  const [preflightOk, setPreflightOk] = useState(false);
  const [words, setWords] = useState([]); // sentence buffer
  const [celebrateKey, setCelebrateKey] = useState(0); // bump to fire confetti
  const stateRef = useRef(state);
  stateRef.current = state;
  const startedAtRef = useRef(0);
  const timerRef = useRef(null);

  // Append an accepted/confirmed word to the sentence buffer, then notify any
  // parent that also wants it (onWord is optional). Fires the celebration.
  const addWord = useCallback(
    (label) => {
      setWords((prev) => [...prev, label]);
      setCelebrateKey((k) => k + 1);
      if (onWord) onWord(label);
    },
    [onWord]
  );

  const beginRecording = useCallback(() => {
    if (!ready || stateRef.current !== "idle") return;
    setResult(null);
    setMessage("");
    setConfirmed(null);
    setFrameCount(0);
    setElapsedMs(0);
    setState("recording");
    startedAtRef.current = performance.now();
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setElapsedMs(performance.now() - startedAtRef.current);
    }, 100);
    startCapture(CAPTURE_FPS, (n) => setFrameCount(n));
  }, [ready, startCapture]);

  // User tapped one of the top-3 chips in a "confirm" result (Section 6.2).
  const onConfirmChoice = useCallback(
    async (label) => {
      setConfirmed(label);
      addWord(label); // confirmed words also enter the sentence buffer
      if (!result?.clip_id) return;
      try {
        await confirmChoice({ clip_id: result.clip_id, chosen_label: label });
      } catch {
        // Non-fatal: the word is still shown; the sample just wasn't logged.
      }
    },
    [result, addWord]
  );

  const endRecordingAndUpload = useCallback(async () => {
    if (stateRef.current !== "recording") return;
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    const frames = stopCapture();
    setState("analysing");

    if (!frames.length) {
      setMessage("No frames captured — please try again.");
      setState("idle");
      return;
    }

    try {
      const data = await recognizeClip({
        frames,
        fps: CAPTURE_FPS,
        signer_id: signerId,
      });
      setResult(data);
      if (data.decision === "reject") {
        setMessage(rejectMessage(data.reject_reason));
      } else {
        setMessage("");
        if (data.decision === "accept" && data.label) {
          addWord(data.label); // accepted words enter the sentence buffer
        }
      }
    } catch (err) {
      setMessage(
        err?.response?.data?.detail
          ? `Server error: ${err.response.data.detail}`
          : "Could not reach the server. Is it running?"
      );
    } finally {
      setState("idle");
    }
  }, [signerId, stopCapture, addWord]);

  // Toggle: start if idle, stop if recording.
  const toggleRecording = useCallback(() => {
    if (stateRef.current === "recording") endRecordingAndUpload();
    else beginRecording();
  }, [beginRecording, endRecordingAndUpload]);

  // Space toggles capture (tap, not hold) — hands-free friendly. Ignore
  // auto-repeat and don't hijack Space while typing in a form field.
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.code !== "Space" || e.repeat) return;
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      e.preventDefault();
      toggleRecording();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [toggleRecording]);

  // Clean up the timer if the component unmounts mid-recording.
  useEffect(() => () => timerRef.current && clearInterval(timerRef.current), []);

  const seconds = (elapsedMs / 1000).toFixed(1);

  return (
    <section className="mx-auto max-w-xl">
      <div className="glass p-5 sm:p-6">
        <div className="mb-4 flex items-center justify-end">
          <StatusBadge state={state} />
        </div>

        {error && (
          <p
            className="mb-4 rounded-xl border border-red-400/30 bg-red-500/15 p-3 text-red-100"
            role="alert"
          >
            {error}
          </p>
        )}

        <div className="relative">
          <VideoPreview videoRef={videoRef} mirrored recording={state === "recording"} />
          {/* Live recording timer, top-right over the preview. */}
          {state === "recording" && (
            <span className="absolute right-3 top-3 z-10 inline-flex items-center gap-1.5 rounded-full bg-black/60 px-3 py-1 font-mono text-sm font-semibold text-white shadow-lg backdrop-blur-sm">
              <span className="h-2 w-2 animate-pulse rounded-full bg-red-400" />
              {seconds}s
            </span>
          )}
          {/* Confetti burst anchored to the preview on each accepted word. */}
          <Celebrate trigger={celebrateKey} />
        </div>

        <PreflightCheck videoRef={videoRef} ready={ready} onReadyChange={setPreflightOk} />
        {ready && !preflightOk && (
          <p className="mt-2 text-xs text-amber-300">
            Adjust lighting/framing until the checks turn green. You can still
            tap to sign anyway.
          </p>
        )}

        {state === "recording" && (
          <p className="mt-2 flex items-center gap-2 text-sm text-slate-300">
            <span className="h-1.5 w-1.5 animate-ping rounded-full bg-red-400" />
            Recording… {frameCount} frames captured
          </p>
        )}

        {/* Tap to start / tap to stop — a toggle, keyboard + touch operable. */}
        <div className="mt-4">
          <GlassButton
            variant={state === "recording" ? "danger" : "brand"}
            disabled={!ready || state === "analysing"}
            onClick={toggleRecording}
            className={`w-full !rounded-2xl px-4 py-4 text-lg ${
              state === "recording" ? "animate-pulse-ring" : ""
            }`}
          >
            {state === "recording" ? (
              <>
                <StopGlyph /> Tap to stop &amp; translate
              </>
            ) : state === "analysing" ? (
              "Analysing…"
            ) : (
              <>
                <RecGlyph /> Tap to sign{" "}
                <kbd className="rounded-md border border-white/20 bg-white/10 px-2 py-0.5 text-sm">
                  Space
                </kbd>
              </>
            )}
          </GlassButton>
          <p className="mt-2 text-center text-xs text-slate-500">
            Tap once to start, sign with both hands free, then tap again to stop.
          </p>
        </div>

        <div className="mt-6 grid min-h-[7rem] place-items-center overflow-hidden rounded-xl border border-white/10 bg-ink-900/50 p-5 text-center">
          {state === "analysing" ? (
            <p className="text-lg font-semibold shimmer-text" aria-live="polite">
              Analysing your sign…
            </p>
          ) : confirmed ? (
            <div className="animate-pop-in">
              <p className="text-xs uppercase tracking-widest text-slate-400">
                Confirmed word
              </p>
              <p
                className="mt-1 animate-gradient-pan bg-gradient-to-r from-emerald-300 via-teal-200 to-cyan-300 bg-[length:200%_auto] bg-clip-text text-5xl font-extrabold text-transparent"
                aria-live="polite"
              >
                {confirmed}
              </p>
            </div>
          ) : result && result.decision === "accept" && result.label ? (
            <div className="animate-pop-in">
              <p className="text-xs uppercase tracking-widest text-slate-400">
                Recognised word
              </p>
              <p
                className="mt-1 animate-gradient-pan bg-gradient-to-r from-emerald-300 via-teal-200 to-cyan-300 bg-[length:200%_auto] bg-clip-text text-5xl font-extrabold text-transparent"
                aria-live="polite"
              >
                {result.label}
              </p>
            </div>
          ) : result && result.decision === "confirm" ? (
            <div className="animate-fade-in" aria-live="polite">
              <p className="mb-3 text-sm font-medium text-amber-200">
                Did you mean one of these? Tap the right word, or sign again.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {result.candidates.map((c, i) => (
                  <GlassButton
                    key={c.label}
                    variant="clear"
                    onClick={() => onConfirmChoice(c.label)}
                    style={{ animationDelay: `${i * 70}ms` }}
                    className="animate-pop-in !px-4 !py-2 text-amber-50"
                  >
                    {c.label}
                    <span className="ml-1 rounded-full bg-black/30 px-1.5 py-0.5 text-xs opacity-90">
                      {Math.round(c.p * 100)}%
                    </span>
                  </GlassButton>
                ))}
              </div>
            </div>
          ) : message ? (
            <p className="animate-fade-in text-lg text-amber-200" aria-live="polite">
              {message}
            </p>
          ) : (
            <p className="text-slate-400">
              Tap to start, sign a word, then tap to stop. The recognised word
              appears here.
            </p>
          )}
        </div>

        <SentenceBox words={words} onClear={() => setWords([])} />
      </div>
    </section>
  );
}

/** Small record-dot glyph for the idle button. */
function RecGlyph() {
  return <span className="inline-block h-3 w-3 rounded-full bg-white/90 shadow-[0_0_8px_2px_rgba(255,255,255,0.6)]" aria-hidden="true" />;
}
/** Small stop-square glyph for the recording button. */
function StopGlyph() {
  return <span className="inline-block h-3 w-3 rounded-[3px] bg-white" aria-hidden="true" />;
}

/** Map a server reject_reason to a short, friendly message. */
function rejectMessage(reason) {
  switch (reason) {
    case "hands_not_visible":
      return "Hands weren't visible — please sign again.";
    case "too_short":
      return "That was too quick — record a little longer.";
    case "too_long":
      return "That was too long — try a shorter clip.";
    case "low_confidence":
      return "Not sure what that was — please sign again.";
    default:
      return "Please sign again.";
  }
}
