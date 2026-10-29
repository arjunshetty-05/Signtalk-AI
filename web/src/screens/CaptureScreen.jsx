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
  const {
    videoRef,
    ready,
    error,
    wideView,
    cameras,
    selectedDeviceId,
    setSelectedDeviceId,
    startCapture,
    stopCapture,
  } = useWebcam();
  const [state, setState] = useState("idle"); // idle | countdown | recording | analysing
  const [captureMode, setCaptureMode] = useState("hold"); // hold | click
  const [countdown, setCountdown] = useState(2);
  const [frameCount, setFrameCount] = useState(0);
  const [result, setResult] = useState(null); // recognize response
  const [message, setMessage] = useState("");
  const [confirmed, setConfirmed] = useState(null); // label the user tapped
  const [preflightOk, setPreflightOk] = useState(false);
  const [words, setWords] = useState([]); // sentence buffer
  const stateRef = useRef(state);
  stateRef.current = state;
  const spaceHeldRef = useRef(false);
  const captureModeRef = useRef(captureMode);
  captureModeRef.current = captureMode;
  const countdownTimerRef = useRef(null);

  // Append an accepted/confirmed word to the sentence buffer, then notify any
  // parent that also wants it (onWord is optional).
  const addWord = useCallback(
    (label) => {
      setWords((prev) => [...prev, label]);
      if (onWord) onWord(label);
    },
    [onWord]
  );

  const startFrameCapture = useCallback(() => {
    if (!ready) return;
    setResult(null);
    setMessage("");
    setConfirmed(null);
    setFrameCount(0);
    setState("recording");
    startCapture(CAPTURE_FPS, (n) => setFrameCount(n));
  }, [ready, startCapture]);

  const beginRecording = useCallback(() => {
    if (stateRef.current !== "idle") return;
    startFrameCapture();
  }, [startFrameCapture]);

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

  const handleClickMode = useCallback(() => {
    if (stateRef.current === "countdown") {
      clearTimeout(countdownTimerRef.current);
      countdownTimerRef.current = null;
      setState("idle");
      return;
    }
    if (stateRef.current === "recording") {
      endRecordingAndUpload();
      return;
    }
    if (!ready || stateRef.current !== "idle") return;
    setResult(null);
    setMessage("");
    setConfirmed(null);
    setFrameCount(0);
    setCountdown(2);
    setState("countdown");
    countdownTimerRef.current = setTimeout(() => {
      setCountdown(1);
      countdownTimerRef.current = setTimeout(() => {
        countdownTimerRef.current = null;
        startFrameCapture();
      }, 1000);
    }, 1000);
  }, [ready, endRecordingAndUpload, startFrameCapture]);

  useEffect(() => () => clearTimeout(countdownTimerRef.current), []);

  // Hold-Space push-to-sign without scrolling or hijacking other controls.
  useEffect(() => {
    const onKeyDown = (e) => {
      if (captureModeRef.current !== "hold") return;
      if (e.code !== "Space") return;
      const target = e.target instanceof Element ? e.target : null;
      if (
        target?.closest('input, textarea, select, [contenteditable="true"]')
      ) {
        return;
      }
      const captureButton = target?.closest("[data-space-capture]");
      if (
        !captureButton &&
        target?.closest('button, a, [role="button"]')
      ) {
        return;
      }
      e.preventDefault();
      if (e.repeat) return;
      spaceHeldRef.current = true;
      beginRecording();
    };
    const onKeyUp = (e) => {
      if (captureModeRef.current !== "hold") return;
      if (e.code !== "Space" || !spaceHeldRef.current) return;
      e.preventDefault();
      spaceHeldRef.current = false;
      endRecordingAndUpload();
    };
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
    };
  }, [beginRecording, endRecordingAndUpload]);

  return (
    <section className="mx-auto max-w-6xl">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
        <div>
          <p className="mb-1 text-xs font-bold uppercase tracking-[0.18em] text-[#8b6a38]">
            Sign to sentence
          </p>
          <h2 className="text-2xl font-bold tracking-tight text-[#1d2c25] sm:text-3xl">
            Sign capture
          </h2>
          <p className="mt-1 text-sm text-[#69776e]">
            Frame your hands, hold to sign, then review the recognized word.
          </p>
        </div>
        <StatusBadge state={state} />
      </div>

      {error && (
        <p className="mb-4 rounded-xl border border-[#f1c6bc] bg-[#fff1ed] p-3 text-[#9b3c2f]" role="alert">
          {error}
        </p>
      )}

      <div className="grid min-w-0 items-start gap-3 sm:grid-cols-[minmax(0,1.05fr)_minmax(260px,0.95fr)] lg:grid-cols-[minmax(0,1.15fr)_minmax(360px,0.85fr)] xl:gap-4">
        <div className="min-w-0 rounded-2xl border border-[#e6e3db] bg-white p-3 shadow-[0_8px_30px_rgba(36,49,40,0.05)] sm:p-4">
          <div className="mb-2 flex min-w-0 flex-col items-start justify-between gap-2 sm:flex-row sm:items-center">
            <div>
              <h3 className="font-semibold text-[#28372f]">Camera preview</h3>
              <p className="mt-0.5 text-xs text-[#7a867e]">Keep both hands inside the frame</p>
            </div>
            {cameras.length > 1 && (
              <label className="flex w-full flex-col items-start gap-1 text-sm text-[#66736b] sm:w-auto sm:flex-row sm:items-center">
                <span className="sr-only">Camera</span>
                <select
                  aria-label="Camera"
                  value={selectedDeviceId}
                  disabled={state !== "idle"}
                  onChange={(event) => setSelectedDeviceId(event.target.value)}
                  className="w-full max-w-full rounded-lg border border-[#dddcd4] bg-[#fbfaf7] px-3 py-2 text-sm text-[#28372f] disabled:opacity-60 sm:max-w-44"
                >
                  {cameras.map((camera, index) => (
                    <option key={camera.deviceId} value={camera.deviceId}>
                      {camera.label || `Camera ${index + 1}`}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          <VideoPreview videoRef={videoRef} mirrored recording={state === "recording"} countdown={state === "countdown" ? countdown : null} />
          {ready && (
            <p className="mt-2 text-xs leading-relaxed text-[#748078]" role="status">
              {wideView
                ? "Camera is set to its widest supported view."
                : "For a wider view, try another camera or move this one farther away."}
            </p>
          )}
          <PreflightCheck videoRef={videoRef} ready={ready} onReadyChange={setPreflightOk} />
          {ready && !preflightOk && (
            <p className="mt-1 text-xs leading-relaxed text-[#916b2c]">
              Adjust lighting and framing for best results. You can still sign if a check is amber.
            </p>
          )}
        </div>

        <div className="min-w-0 space-y-3">
          <div className="rounded-2xl border border-[#e6e3db] bg-white p-3 shadow-[0_8px_30px_rgba(36,49,40,0.05)] sm:p-4">
            <div className="mb-3 flex rounded-xl bg-[#f3f4ef] p-1" role="group" aria-label="Recording mode">
              <button type="button" aria-pressed={captureMode === "hold"} disabled={state !== "idle"} onClick={() => setCaptureMode("hold")} className={`flex-1 rounded-lg px-3 py-2 text-sm font-semibold transition ${captureMode === "hold" ? "bg-white text-[#1c5b46] shadow-sm" : "text-[#69776e] hover:text-[#28372f]"}`}>Hold to record</button>
              <button type="button" aria-pressed={captureMode === "click"} disabled={state !== "idle"} onClick={() => setCaptureMode("click")} className={`flex-1 rounded-lg px-3 py-2 text-sm font-semibold transition ${captureMode === "click" ? "bg-white text-[#1c5b46] shadow-sm" : "text-[#69776e] hover:text-[#28372f]"}`}>Click to record</button>
            </div>
            <p className="mb-2 text-sm text-[#69776e]">
              {captureMode === "click" ? "Click once, get into position during the 2-second countdown, then click again when you’re done." : "Hold the key or button while signing; release when you’re done."}
            </p>
            <button
              type="button"
              disabled={!ready || state === "analysing"}
              data-space-capture
              onClick={captureMode === "click" ? handleClickMode : undefined}
              onMouseDown={captureMode === "hold" ? beginRecording : undefined}
              onMouseUp={captureMode === "hold" ? endRecordingAndUpload : undefined}
              onMouseLeave={captureMode === "hold" ? () => state === "recording" && endRecordingAndUpload() : undefined}
              onTouchStart={captureMode === "hold" ? (e) => { e.preventDefault(); beginRecording(); } : undefined}
              onTouchEnd={captureMode === "hold" ? (e) => { e.preventDefault(); endRecordingAndUpload(); } : undefined}
              className={`w-full rounded-xl px-4 py-3 text-base font-bold text-white shadow-sm transition hover:-translate-y-0.5 focus-visible:ring disabled:cursor-not-allowed disabled:opacity-50 ${
                state === "recording"
                  ? "bg-[#b64a3a] hover:bg-[#a43d30]"
                  : "bg-[#1c5b46] hover:bg-[#164b39]"
              }`}
            >
              {captureMode === "click" ? (state === "countdown" ? `Cancel countdown · ${countdown}` : state === "recording" ? "Finish and recognize" : "Click to start signing") : (state === "recording" ? "Release to recognize" : <>Hold <kbd className="mx-1 rounded-md border border-white/25 bg-white/10 px-2 py-1 text-sm">Space</kbd> to sign</>)}
            </button>
            <div className="mt-2 flex min-h-6 items-center justify-between text-sm">
              <span className="text-[#69776e]" role="status" aria-live="polite">
                {state === "countdown"
                  ? `Get ready · recording starts in ${countdown}…`
                  : state === "recording"
                  ? `Recording · ${frameCount} frames`
                  : state === "analysing"
                    ? "Analyzing your sign…"
                    : "Ready when you are"}
              </span>
              {state === "recording" && (
                <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-[#c94e3f]" aria-hidden="true" />
              )}
            </div>
          </div>

          <div className="min-h-[5rem] rounded-2xl border border-[#e6e3db] bg-white p-4 shadow-[0_8px_30px_rgba(36,49,40,0.05)]">
            {state === "analysing" ? (
              <p className="text-lg font-medium text-[#916b2c]" aria-live="polite">
                Analyzing…
              </p>
            ) : confirmed ? (
              <>
                <p className="text-xs font-bold uppercase tracking-[0.14em] text-[#748078]">Confirmed word</p>
                <p className="mt-1 text-2xl font-bold tracking-tight text-[#1c5b46]" aria-live="polite">{confirmed}</p>
              </>
            ) : result && result.decision === "accept" && result.label ? (
              <>
                <p className="text-xs font-bold uppercase tracking-[0.14em] text-[#748078]">Recognized word</p>
                <p className="mt-1 text-2xl font-bold tracking-tight text-[#1c5b46]" aria-live="polite">{result.label}</p>
              </>
            ) : result && result.decision === "confirm" ? (
              <div aria-live="polite">
                <p className="mb-2 text-sm font-medium text-[#8b6425]">Choose the closest match, or sign again.</p>
                <div className="flex flex-wrap gap-2">
                  {result.candidates.map((c) => (
                    <button
                      key={c.label}
                      type="button"
                      onClick={() => onConfirmChoice(c.label)}
                      className="rounded-full border border-[#e7d4ae] bg-[#fff8e9] px-4 py-2 font-semibold text-[#73531e] transition hover:bg-[#f7e9cc] focus-visible:ring"
                    >
                      {c.label}<span className="ml-2 text-xs opacity-70">{Math.round(c.p * 100)}%</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : message ? (
              <p className="text-base text-[#9b5532]" aria-live="polite">{message}</p>
            ) : (
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.14em] text-[#748078]">Your word</p>
                <p className="mt-2 text-sm leading-relaxed text-[#69776e]">
                  The recognized sign will appear here after you finish recording.
                </p>
              </div>
            )}
          </div>
          <SentenceBox words={words} onClear={() => setWords([])} />
        </div>
      </div>
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
