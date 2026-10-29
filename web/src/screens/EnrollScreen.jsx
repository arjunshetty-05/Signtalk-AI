import { useCallback, useEffect, useRef, useState } from "react";
import { getVocab, enrollStart, enrollClip, enrollProgress } from "../api.js";
import VideoPreview from "../components/VideoPreview.jsx";
import StatusBadge from "../components/StatusBadge.jsx";
import { openWideWebcam } from "../camera.js";

// Target number of repetitions to record per sign during enrollment.
const TARGET_REPS = 5;

/**
 * Enrollment recording screen (Section 6.5, Phase-1 subset).
 *
 * Pick a signer, start a session, choose a sign from /api/vocab, then record N
 * reps with a live counter. Each rep is recorded as a WebM clip, base64-encoded
 * and POSTed to /api/enroll/clip; the counter is refreshed from
 * /api/enroll/progress.
 *
 * @param {{signerId: string}} props  the current guest signer id
 */
export default function EnrollScreen({ signerId }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const spaceHeldRef = useRef(false);

  const [camError, setCamError] = useState(null);
  const [wideView, setWideView] = useState(null);
  const [vocab, setVocab] = useState([]);
  const [sign, setSign] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [state, setState] = useState("idle"); // idle | recording | analysing
  const [counts, setCounts] = useState({});
  const [message, setMessage] = useState("");

  // Open the webcam once for this screen.
  useEffect(() => {
    let cancelled = false;
    async function open() {
      try {
        const { stream, wideView: zoomedOut } = await openWideWebcam({
          width: 640,
          height: 480,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        setWideView(zoomedOut);
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        if (!cancelled) setCamError(err?.message || "Could not access the webcam.");
      }
    }
    open();
    return () => {
      cancelled = true;
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // Load the vocabulary.
  useEffect(() => {
    let cancelled = false;
    getVocab()
      .then((items) => {
        if (cancelled) return;
        setVocab(items);
        if (items.length) setSign(items[0].label);
      })
      .catch(() => !cancelled && setMessage("Could not load the vocabulary."));
    return () => {
      cancelled = true;
    };
  }, []);

  // Start (or restart) an enrollment session for this signer.
  const startSession = useCallback(async () => {
    try {
      const data = await enrollStart(signerId);
      setSessionId(data.session_id);
      const prog = await enrollProgress(signerId);
      setCounts(prog.counts || {});
      setMessage("Session started. Pick a sign and record your reps.");
    } catch {
      setMessage("Could not start a session. Is the server running?");
    }
  }, [signerId]);

  const recordedCount = counts[sign] || 0;

  const stopAndUpload = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") return;

    const done = new Promise((resolve) => {
      recorder.onstop = resolve;
    });
    recorder.stop();
    await done;

    setState("analysing");
    const blob = new Blob(chunksRef.current, { type: "video/webm" });
    chunksRef.current = [];

    try {
      const base64 = await blobToBase64(blob);
      const data = await enrollClip({
        signer_id: signerId,
        session_id: sessionId,
        sign,
        clip: base64,
      });
      setCounts((prev) => ({ ...prev, [sign]: data.count }));
      setMessage(`Saved rep ${data.count} for "${sign}".`);
    } catch (err) {
      setMessage(
        err?.response?.data?.detail
          ? `Server error: ${err.response.data.detail}`
          : "Could not upload the clip."
      );
    } finally {
      setState("idle");
    }
  }, [signerId, sessionId, sign]);

  const startRecording = useCallback(() => {
    if (!sessionId) {
      setMessage("Start a session first.");
      return;
    }
    if (!sign || state !== "idle" || !streamRef.current) return;

    chunksRef.current = [];
    const recorder = new MediaRecorder(streamRef.current, {
      mimeType: "video/webm",
    });
    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorderRef.current = recorder;
    recorder.start();
    setState("recording");
    setMessage("");
  }, [sessionId, sign, state]);

  // Hold-Space push-to-record without scrolling or hijacking other controls.
  useEffect(() => {
    const onKeyDown = (e) => {
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
      startRecording();
    };
    const onKeyUp = (e) => {
      if (e.code !== "Space" || !spaceHeldRef.current) return;
      e.preventDefault();
      spaceHeldRef.current = false;
      stopAndUpload();
    };
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
    };
  }, [startRecording, stopAndUpload]);

  return (
    <section className="mx-auto max-w-4xl">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
        <div>
          <p className="mb-1 text-xs font-bold uppercase tracking-[0.18em] text-[#8b6a38]">Improve the vocabulary</p>
          <h2 className="text-2xl font-bold tracking-tight text-[#1d2c25] sm:text-3xl">Enrollment</h2>
          <p className="mt-1 text-sm text-[#69776e]">Record multiple examples of a sign for future model training.</p>
        </div>
        <StatusBadge state={state} />
      </div>

      {camError && (
        <p className="mb-4 rounded-xl border border-[#f1c6bc] bg-[#fff1ed] p-3 text-[#9b3c2f]" role="alert">
          {camError}
        </p>
      )}

      <div className="mb-3 grid gap-3 rounded-xl border border-[#e6e3db] bg-white p-3 shadow-[0_4px_18px_rgba(36,49,40,0.04)] sm:grid-cols-2 sm:p-4">
        <div>
          <label className="mb-1.5 block text-sm font-medium text-[#52635a]" htmlFor="signer">
            Signer
          </label>
          <input
            id="signer"
            value={signerId}
            readOnly
            className="w-full rounded-lg border border-[#e2e1d9] bg-[#f8f7f3] px-3 py-2.5 text-[#56645b]"
          />
        </div>
        <div>
          <label className="mb-1.5 block text-sm font-medium text-[#52635a]" htmlFor="sign">
            Sign to record
          </label>
          <select
            id="sign"
            value={sign}
            onChange={(e) => setSign(e.target.value)}
            className="w-full rounded-lg border border-[#d8ddd7] bg-white px-3 py-2.5 text-[#26362e]"
          >
            {vocab.map((v) => (
              <option key={v.label} value={v.label}>
                {v.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-2 rounded-xl border border-[#e6e3db] bg-[#fbfaf7] p-2.5">
        <button
          type="button"
          onClick={startSession}
          className="rounded-lg bg-[#1c5b46] px-4 py-2.5 font-semibold text-white hover:bg-[#164b39] focus-visible:ring"
        >
          {sessionId ? "Restart session" : "Start session"}
        </button>
        <span className="text-sm text-[#68766e]" aria-live="polite">
          Reps for "{sign}": <strong>{recordedCount}</strong> / {TARGET_REPS}
        </span>
      </div>

      <VideoPreview videoRef={videoRef} mirrored recording={state === "recording"} />
      {wideView !== null && (
        <p className="mt-2 text-sm text-[#748078]" role="status">
          {wideView
            ? "Camera is set to its widest supported view."
            : "For a wider view, move the camera farther away; this camera may not support zoom-out."}
        </p>
      )}

      <div className="mt-3">
        <button
          type="button"
          disabled={!sessionId}
          data-space-capture
          onMouseDown={startRecording}
          onMouseUp={stopAndUpload}
          onMouseLeave={() => state === "recording" && stopAndUpload()}
          onTouchStart={(e) => {
            e.preventDefault();
            startRecording();
          }}
          onTouchEnd={(e) => {
            e.preventDefault();
            stopAndUpload();
          }}
          className="w-full rounded-xl bg-[#1c5b46] px-4 py-3 text-base font-bold text-white hover:bg-[#164b39] focus-visible:ring disabled:opacity-50"
        >
          Hold <kbd className="rounded bg-black/40 px-2">Space</kbd> (or this button) to record a rep
        </button>
      </div>

      <div className="mt-3 min-h-[2.5rem] rounded-xl border border-[#e6e3db] bg-white p-3 text-center">
        {state === "analysing" ? (
          <p className="text-[#916b2c]" aria-live="polite">
            Saving clip…
          </p>
        ) : (
          <p className="text-[#68766e]" aria-live="polite">
            {message ||
              (recordedCount >= TARGET_REPS
                ? `Done — ${recordedCount} reps recorded for "${sign}".`
                : "Record each rep by holding Space.")}
          </p>
        )}
      </div>
    </section>
  );
}

/** Read a Blob into a bare base64 string (no data: prefix). */
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}
