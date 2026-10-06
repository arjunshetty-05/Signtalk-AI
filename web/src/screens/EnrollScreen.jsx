import { useCallback, useEffect, useRef, useState } from "react";
import { getVocab, enrollStart, enrollClip, enrollProgress } from "../api.js";
import VideoPreview from "../components/VideoPreview.jsx";
import StatusBadge from "../components/StatusBadge.jsx";

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

  const [camError, setCamError] = useState(null);
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
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
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

  // Hold-Space push-to-record (same gesture as capture). Ignore when typing or
  // interacting with form controls.
  useEffect(() => {
    const typingTarget = (t) =>
      t === "INPUT" || t === "TEXTAREA" || t === "SELECT";
    const onKeyDown = (e) => {
      if (e.code !== "Space" || e.repeat || typingTarget(e.target?.tagName)) return;
      e.preventDefault();
      startRecording();
    };
    const onKeyUp = (e) => {
      if (e.code !== "Space" || typingTarget(e.target?.tagName)) return;
      e.preventDefault();
      stopAndUpload();
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [startRecording, stopAndUpload]);

  return (
    <section className="mx-auto max-w-xl">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-bold">Enrollment</h2>
        <StatusBadge state={state} />
      </div>

      {camError && (
        <p className="mb-4 rounded bg-red-900 p-3 text-red-100" role="alert">
          {camError}
        </p>
      )}

      <div className="mb-4 grid gap-3 sm:grid-cols-2">
        <div>
          <label className="mb-1 block text-sm text-gray-300" htmlFor="signer">
            Signer
          </label>
          <input
            id="signer"
            value={signerId}
            readOnly
            className="w-full rounded bg-gray-800 px-3 py-2 text-gray-100"
          />
        </div>
        <div>
          <label className="mb-1 block text-sm text-gray-300" htmlFor="sign">
            Sign to record
          </label>
          <select
            id="sign"
            value={sign}
            onChange={(e) => setSign(e.target.value)}
            className="w-full rounded bg-gray-800 px-3 py-2 text-gray-100"
          >
            {vocab.map((v) => (
              <option key={v.label} value={v.label}>
                {v.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mb-4 flex items-center gap-3">
        <button
          type="button"
          onClick={startSession}
          className="rounded bg-indigo-600 px-4 py-2 font-semibold text-white hover:bg-indigo-500 focus-visible:ring"
        >
          {sessionId ? "Restart session" : "Start session"}
        </button>
        <span className="text-sm text-gray-300" aria-live="polite">
          Reps for "{sign}": <strong>{recordedCount}</strong> / {TARGET_REPS}
        </span>
      </div>

      <VideoPreview videoRef={videoRef} mirrored recording={state === "recording"} />

      <div className="mt-4">
        <button
          type="button"
          disabled={!sessionId}
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
          className="w-full rounded-lg bg-blue-600 px-4 py-4 text-lg font-semibold text-white hover:bg-blue-500 focus-visible:ring disabled:opacity-50"
        >
          Hold <kbd className="rounded bg-black/40 px-2">Space</kbd> (or this button) to record a rep
        </button>
      </div>

      <div className="mt-4 min-h-[3rem] rounded-lg border border-gray-700 p-3 text-center">
        {state === "analysing" ? (
          <p className="text-amber-300" aria-live="polite">
            Saving clip…
          </p>
        ) : (
          <p className="text-gray-300" aria-live="polite">
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
