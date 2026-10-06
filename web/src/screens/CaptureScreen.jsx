import { useCallback, useEffect, useRef, useState } from "react";
import { useWebcam } from "../hooks/useWebcam.js";
import { recognizeClip, confirmChoice } from "../api.js";
import VideoPreview from "../components/VideoPreview.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import PreflightCheck from "../components/PreflightCheck.jsx";
import SentenceBox from "../components/SentenceBox.jsx";

// Capture frame rate for the base64 JPEG batch sent to /api/recognize. The
// server resamples to config `sequence_length`, so this only needs to be a
// sensible real-time rate, not an exact match.
const CAPTURE_FPS = 15;

/**
 * Push-to-sign capture screen (record-then-recognise, Section 6.1).
 *
 * Hold Space to record a whole clip, release to stop and upload the UN-mirrored
 * frame batch to /api/recognize, then show the returned label (or the reject
 * reason). The capture state (idle / recording / analysing) is always visible.
 *
 * @param {{signerId: string, onWord?: (label: string) => void}} props
 *   `onWord` is called with each accepted/confirmed word so the parent can
 *   collect them into the sentence buffer.
 */
export default function CaptureScreen({ signerId, onWord }) {
  const { videoRef, ready, error, startCapture, stopCapture } = useWebcam();
  const [state, setState] = useState("idle"); // idle | recording | analysing
  const [frameCount, setFrameCount] = useState(0);
  const [result, setResult] = useState(null); // recognize response
  const [message, setMessage] = useState("");
  const [confirmed, setConfirmed] = useState(null); // label the user tapped
  const [preflightOk, setPreflightOk] = useState(false);
  const [words, setWords] = useState([]); // sentence buffer
  const stateRef = useRef(state);
  stateRef.current = state;

  // Append an accepted/confirmed word to the sentence buffer, then notify any
  // parent that also wants it (onWord is optional).
  const addWord = useCallback(
    (label) => {
      setWords((prev) => [...prev, label]);
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
    setState("recording");
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

  // Hold-Space push-to-sign. Ignore auto-repeat keydown events.
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.code !== "Space" || e.repeat) return;
      // Don't hijack Space while typing in a form field.
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      e.preventDefault();
      beginRecording();
    };
    const onKeyUp = (e) => {
      if (e.code !== "Space") return;
      const tag = e.target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      e.preventDefault();
      endRecordingAndUpload();
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [beginRecording, endRecordingAndUpload]);

  return (
    <section className="mx-auto max-w-xl">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-bold">Sign capture</h2>
        <StatusBadge state={state} />
      </div>

      {error && (
        <p className="mb-4 rounded bg-red-900 p-3 text-red-100" role="alert">
          {error}
        </p>
      )}

      <VideoPreview videoRef={videoRef} mirrored recording={state === "recording"} />

      <PreflightCheck videoRef={videoRef} ready={ready} onReadyChange={setPreflightOk} />
      {ready && !preflightOk && (
        <p className="mt-1 text-xs text-amber-300">
          Adjust lighting/framing until the checks turn green. You can still
          hold Space to sign anyway.
        </p>
      )}

      {state === "recording" && (
        <p className="mt-2 text-sm text-gray-300">Captured {frameCount} frames…</p>
      )}

      {/* Keyboard-first, but also mouse/touch operable via a button. */}
      <div className="mt-4">
        <button
          type="button"
          disabled={!ready}
          onMouseDown={beginRecording}
          onMouseUp={endRecordingAndUpload}
          onMouseLeave={() => state === "recording" && endRecordingAndUpload()}
          onTouchStart={(e) => {
            e.preventDefault();
            beginRecording();
          }}
          onTouchEnd={(e) => {
            e.preventDefault();
            endRecordingAndUpload();
          }}
          className="w-full rounded-lg bg-blue-600 px-4 py-4 text-lg font-semibold text-white hover:bg-blue-500 focus-visible:ring disabled:opacity-50"
        >
          Hold <kbd className="rounded bg-black/40 px-2">Space</kbd> (or hold this button) to sign
        </button>
      </div>

      <div className="mt-6 min-h-[6rem] rounded-lg border border-gray-700 p-4 text-center">
        {state === "analysing" ? (
          <p className="text-lg text-amber-300" aria-live="polite">
            Analysing…
          </p>
        ) : confirmed ? (
          <>
            <p className="text-sm text-gray-400">Confirmed word</p>
            <p className="text-4xl font-bold text-green-400" aria-live="polite">
              {confirmed}
            </p>
          </>
        ) : result && result.decision === "accept" && result.label ? (
          <>
            <p className="text-sm text-gray-400">Recognised word</p>
            <p className="text-4xl font-bold text-green-400" aria-live="polite">
              {result.label}
            </p>
          </>
        ) : result && result.decision === "confirm" ? (
          <div aria-live="polite">
            <p className="mb-2 text-sm text-amber-300">
              Did you mean one of these? Tap the right word, or sign again.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {result.candidates.map((c) => (
                <button
                  key={c.label}
                  type="button"
                  onClick={() => onConfirmChoice(c.label)}
                  className="rounded-full bg-amber-600 px-4 py-2 font-semibold text-white hover:bg-amber-500 focus-visible:ring"
                >
                  {c.label}
                  <span className="ml-2 text-xs opacity-80">
                    {Math.round(c.p * 100)}%
                  </span>
                </button>
              ))}
            </div>
          </div>
        ) : message ? (
          <p className="text-lg text-amber-300" aria-live="polite">
            {message}
          </p>
        ) : (
          <p className="text-gray-400">
            Hold Space, sign a word, then release. The recognised word appears here.
          </p>
        )}
      </div>

      <SentenceBox words={words} onClear={() => setWords([])} />
    </section>
  );
}

/** Map a server reject_reason to a short, friendly message. */
function rejectMessage(reason) {
  switch (reason) {
    case "hands_not_visible":
      return "Hands weren't visible — please sign again.";
    case "too_short":
      return "That was too quick — hold Space a little longer.";
    case "too_long":
      return "That was too long — try a shorter clip.";
    case "low_confidence":
      return "Not sure what that was — please sign again.";
    default:
      return "Please sign again.";
  }
}
