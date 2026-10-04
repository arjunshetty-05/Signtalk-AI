// WebcamView.jsx — SignTalk AI web dashboard
//
// Captures frames via react-webcam at ~20fps and streams them out via the
// sendFrame callback (from useGestureSocket, owned by App.jsx so other
// panels like AnalyticsPanel can share the same connection's data). Renders
// a live label overlay that animates in on each new (sparse, debounced)
// event and lingers until either a new label arrives or LABEL_TIMEOUT_MS
// passes — it does NOT re-trigger the pulse animation on every render/frame.
//
// Also connects to /ws/speech and shows live partial/final transcripts in
// a small separate overlay.
//
// Deliberately NOT using react-webcam's `mirrored` prop: it flips
// getScreenshot()'s captured pixels too, not just the CSS preview — so
// every frame sent to the classifier would be horizontally mirrored
// relative to the INCLUDE training videos (recorded unmirrored), silently
// hurting recognition on any directionally-asymmetric sign.

import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { motion, AnimatePresence } from "framer-motion";
import { WS_BASE_URL } from "../firebase.js";
import { apiClient, useAuth } from "../context/AuthProvider.jsx";

const CAPTURE_FPS = 20;
const LABEL_TIMEOUT_MS = 4000;
const SPEECH_RECONNECT_DELAY_MS = 2000;
// Downscaled capture width for demo-clip frames sent to /pose/classify-clip
// — MoveNet resizes to its own fixed input size regardless, so sending the
// clip at full 1920x1080 just inflates the request payload and per-frame
// inference time for no accuracy gain.
const DEMO_CAPTURE_WIDTH = 480;
// How long a "Record sign" capture runs — matches roughly the average
// INCLUDE clip duration, long enough to complete one sign.
const RECORD_DURATION_MS = 3000;
// Below this, a "Record sign" result is flagged as uncertain rather than
// shown as if it were reliable. Data-backed (see api/pose/service.py's
// MIN_EMIT_CONFIDENCE comment): wrong guesses across the 262-class model
// almost never exceed ~58% confidence, correct ones are usually 75%+.
const LOW_CONFIDENCE_THRESHOLD = 0.5;

// "Record sign": captures RECORD_DURATION_MS of the user's OWN live webcam,
// then submits it to the same /pose/classify-clip whole-clip pipeline the
// demo clips use, instead of streaming through /ws/gesture. This exists
// because testing showed the model itself is genuinely capable across the
// full vocabulary (94.5% correct on a 200-example held-out sample via
// whole-clip classification) — the live-streaming failure mode is
// specifically the continuously-sliding window seeing incomplete gesture
// motion, not the model not knowing enough signs. Recording a short clip
// and classifying it whole sidesteps that, using the user's real camera
// instead of a pre-recorded file — so any of the 262 trained signs can be
// attempted, not just the 6 demo clips. Not guaranteed correct the way the
// verified demo clips are (still subject to the same live/studio domain
// gap and live signing-style variance), but a real shot at the full
// vocabulary instead of none.

// Fallback demo path: replays a known-good INCLUDE clip (real recorded sign,
// already verified to classify correctly) instead of the live webcam, so a
// demo doesn't depend on live camera/lighting/signing conditions. Six clips
// so the demo isn't just one fixed example — each verified independently
// (POST /pose/classify-clip, downscaled+JPEG-compressed like the browser
// actually sends): Summer 99.1%, Spring 99.4%, Winter 95.7%, Fall 99.7%,
// Monsoon 100.0%, Season 98.1%.
//
// Frames are NOT streamed through the live /ws/gesture pipeline. That
// pipeline runs a continuously-sliding 30-frame trailing window, built for
// open-ended live input — for a short ~3s clip, the window sees partial/
// transitional motion before it ever holds the complete gesture, which
// produced a flapping sequence of wrong labels in testing (confirmed: the
// same clip that classifies correctly offline gave "tall", "warm", "old" in
// quick succession over /ws/gesture). Instead, every frame of the clip is
// collected client-side and POSTed as one batch to /pose/classify-clip,
// which mirrors the *training* preprocessing — extract every frame, smooth,
// then uniformly resample the whole clip to 30 frames — so it reproduces
// the accuracy verified offline instead of the live streaming approximation.
const DEMO_CLIPS = [
  { id: "summer", label: "Summer", src: "/demo/summer.mov" },
  { id: "spring", label: "Spring", src: "/demo/spring.mov" },
  { id: "winter", label: "Winter", src: "/demo/winter.mov" },
  { id: "fall", label: "Fall", src: "/demo/fall.mov" },
  { id: "monsoon", label: "Monsoon", src: "/demo/monsoon.mov" },
  { id: "season", label: "Season", src: "/demo/season.mov" },
];

// Mirrors useGestureSocket's reconnect pattern — without onclose/onerror
// handling, once this socket closed for any reason (server restart, network
// blip) the transcript overlay would silently stop updating for the rest of
// the session.
function useSpeechSocket(token) {
  const [transcript, setTranscript] = useState(null); // {text, is_final}
  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(true);

  const connect = useCallback(() => {
    if (!token) return;

    const socket = new WebSocket(`${WS_BASE_URL}/ws/speech?token=${encodeURIComponent(token)}`);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      try {
        setTranscript(JSON.parse(event.data));
      } catch {
        /* ignore malformed frame */
      }
    };

    socket.onclose = () => {
      if (shouldReconnectRef.current) {
        reconnectTimerRef.current = setTimeout(connect, SPEECH_RECONNECT_DELAY_MS);
      }
    };

    socket.onerror = () => socket.close();
  }, [token]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();
    return () => {
      shouldReconnectRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, [connect]);

  return transcript;
}

export default function WebcamView({ latestLabel, connected, sendFrame, onDemoResult, onDemoReset }) {
  const { token } = useAuth();
  const webcamRef = useRef(null);
  const demoVideoRef = useRef(null);
  const demoCanvasRef = useRef(null);
  const demoFramesRef = useRef([]); // collected base64 JPEGs for the current playthrough
  const transcript = useSpeechSocket(token);
  const [labelVisible, setLabelVisible] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  // Mirrors demoMode for the async classify-clip response handler below —
  // that handler is a stable useCallback closure, so it needs a ref (not
  // the state value) to know whether the user is STILL in demo mode by the
  // time the POST resolves, not whether they were when the request started.
  const demoModeRef = useRef(false);
  useEffect(() => {
    demoModeRef.current = demoMode;
  }, [demoMode]);
  // "playing" (clip running, collecting frames) -> "processing" (clip ended,
  // waiting on POST /pose/classify-clip) -> "recognized" or "no-result"
  // (request failed, or returned nothing usable). Distinct from the raw
  // <video> playback state so the UI can say something concrete ("Analyzing
  // sign...") instead of just replaying silently — a bare looping clip with
  // no feedback reads as "nothing is happening" to someone watching a demo.
  const [demoStatus, setDemoStatus] = useState("idle");
  const [replayTick, setReplayTick] = useState(0);
  const [selectedClipId, setSelectedClipId] = useState(DEMO_CLIPS[0].id);
  const selectedClip = DEMO_CLIPS.find((c) => c.id === selectedClipId) ?? DEMO_CLIPS[0];

  // "Record sign" — same idle -> recording -> processing -> recognized/
  // no-result shape as the demo-clip flow, but sourced from the user's own
  // live camera for RECORD_DURATION_MS instead of a pre-recorded file.
  const [recordStatus, setRecordStatus] = useState("idle");
  const [recordSecondsLeft, setRecordSecondsLeft] = useState(0);
  const [recordConfidence, setRecordConfidence] = useState(null);
  const recordFramesRef = useRef([]);

  const startDemo = useCallback((clipId) => {
    if (clipId) setSelectedClipId(clipId);
    setDemoMode(true);
    setDemoStatus("playing");
    setRecordStatus("idle"); // leaving live view — the record indicator is no longer relevant
    demoFramesRef.current = [];
    setReplayTick((t) => t + 1); // forces the <video> to remount and play from frame 0
  }, []);

  const stopDemo = useCallback(() => {
    setDemoMode(false);
    setDemoStatus("idle");
    onDemoReset?.(); // clears any lingering demo label/sentence so it doesn't show over the live feed
  }, [onDemoReset]);

  const handleDemoEnded = useCallback(async () => {
    setDemoStatus("processing");
    const frames = demoFramesRef.current;
    if (frames.length < 2) {
      setDemoStatus("no-result");
      return;
    }
    try {
      const { data } = await apiClient.post("/pose/classify-clip", { frames });
      // If the user switched back to live camera (or started a different
      // clip) while this request was in flight, don't apply a now-stale
      // result — checked via ref, not the `demoMode` closure value, since
      // this callback doesn't get recreated when demoMode changes.
      if (!demoModeRef.current) return;
      onDemoResult?.(data);
      setDemoStatus("recognized");
    } catch {
      if (demoModeRef.current) setDemoStatus("no-result");
    }
  }, [onDemoResult]);

  // Stream frames at ~CAPTURE_FPS from the live webcam. (Demo-clip capture
  // is handled by a separate effect below.)
  useEffect(() => {
    if (demoMode) return undefined;
    const interval = setInterval(() => {
      const screenshot = webcamRef.current?.getScreenshot();
      if (screenshot) sendFrame(screenshot);
    }, 1000 / CAPTURE_FPS);
    return () => clearInterval(interval);
  }, [sendFrame, demoMode]);

  const handleRecordFinished = useCallback(async () => {
    setRecordStatus("processing");
    const frames = recordFramesRef.current;
    if (frames.length < 2) {
      setRecordStatus("no-result");
      return;
    }
    try {
      const { data } = await apiClient.post("/pose/classify-clip", { frames });
      // Same staleness guard as the demo-clip flow: don't apply a result
      // for a recording the user has since moved on from (e.g. switched to
      // demo mode while this POST was in flight).
      if (demoModeRef.current) return;
      onDemoResult?.(data);
      // Unlike the demo clips (always verified 84%+), a live recording can
      // legitimately land on a near-guess — e.g. 17.8% confidence across
      // 262 classes is barely above chance. classify-clip has no
      // confidence gate (a single explicit attempt should always return
      // *something*, unlike the continuous live stream), so the frontend
      // has to flag low confidence itself instead of presenting a fully-
      // formed sentence as if it were reliable. `data.low_confidence` is a
      // different signal (NLP paraphrase word-overlap, not gesture
      // confidence) so it's not usable here — check confidence directly.
      setRecordConfidence(data.confidence);
      setRecordStatus(data.confidence < LOW_CONFIDENCE_THRESHOLD ? "uncertain" : "recognized");
    } catch {
      if (!demoModeRef.current) setRecordStatus("no-result");
    }
  }, [onDemoResult]);

  const startRecording = useCallback(() => {
    recordFramesRef.current = [];
    setRecordStatus("recording");
    setRecordSecondsLeft(Math.ceil(RECORD_DURATION_MS / 1000));
    setRecordConfidence(null);
  }, []);

  // Countdown display only — cosmetic, decoupled from the actual capture
  // timing below so a dropped tick here can't desync what's collected.
  useEffect(() => {
    if (recordStatus !== "recording") return undefined;
    const interval = setInterval(() => {
      setRecordSecondsLeft((s) => Math.max(0, s - 1));
    }, 1000);
    return () => clearInterval(interval);
  }, [recordStatus]);

  // Live-recording capture: same requestVideoFrameCallback approach as the
  // demo clips (see the comment on RECORD_DURATION_MS above for why this
  // whole-clip method is used instead of streaming), but reading from
  // react-webcam's underlying <video> element and bounded by a fixed
  // duration timer instead of the clip's own `onEnded` event.
  useEffect(() => {
    if (recordStatus !== "recording") return undefined;
    const video = webcamRef.current?.video;
    const canvas = demoCanvasRef.current;
    if (!video || !canvas || typeof video.requestVideoFrameCallback !== "function") {
      setRecordStatus("no-result");
      return undefined;
    }

    let cancelled = false;
    let handle;
    let frameIndex = 0;
    const onFrame = () => {
      if (cancelled) return;
      frameIndex += 1;
      if (frameIndex % 2 === 0) {
        const scale = DEMO_CAPTURE_WIDTH / video.videoWidth;
        canvas.width = DEMO_CAPTURE_WIDTH;
        canvas.height = Math.round(video.videoHeight * scale);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        recordFramesRef.current.push(canvas.toDataURL("image/jpeg", 0.8));
      }
      handle = video.requestVideoFrameCallback(onFrame);
    };
    handle = video.requestVideoFrameCallback(onFrame);

    const timer = setTimeout(() => {
      cancelled = true;
      if (handle) video.cancelVideoFrameCallback(handle);
      handleRecordFinished();
    }, RECORD_DURATION_MS);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      if (handle) video.cancelVideoFrameCallback(handle);
    };
  }, [recordStatus, handleRecordFinished]);

  // Demo-clip capture: collect every rendered frame into demoFramesRef,
  // driven by requestVideoFrameCallback rather than a setInterval timer —
  // a fixed-interval timer starts sampling immediately on play(), but
  // browsers have a brief decode/startup delay before the video actually
  // begins advancing, so the first several "frames" on a timer are often
  // near-duplicates of frame 0. requestVideoFrameCallback only fires once
  // per actually-rendered frame, so every captured frame is a genuine,
  // unique step through the clip. Frames are collected here and sent as one
  // batch on `onEnded` (see handleDemoEnded), not streamed live — see the
  // comment on DEMO_CLIP_SRC above for why.
  useEffect(() => {
    if (!demoMode || demoStatus !== "playing") return undefined;
    const video = demoVideoRef.current;
    const canvas = demoCanvasRef.current;
    if (!video || !canvas || typeof video.requestVideoFrameCallback !== "function") {
      return undefined;
    }

    let cancelled = false;
    let handle;
    let frameIndex = 0;
    const onFrame = () => {
      if (cancelled) return;
      frameIndex += 1;
      // Every other rendered frame is plenty — the server resamples the
      // whole collected sequence down to 30 frames anyway (see
      // /pose/classify-clip), so extra density here only adds MoveNet
      // inference time per frame without adding accuracy. Keeps the
      // "Processing..." wait short.
      if (frameIndex % 2 === 0) {
        const scale = DEMO_CAPTURE_WIDTH / video.videoWidth;
        canvas.width = DEMO_CAPTURE_WIDTH;
        canvas.height = Math.round(video.videoHeight * scale);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        demoFramesRef.current.push(canvas.toDataURL("image/jpeg", 0.8));
      }
      handle = video.requestVideoFrameCallback(onFrame);
    };
    handle = video.requestVideoFrameCallback(onFrame);

    return () => {
      cancelled = true;
      if (handle) video.cancelVideoFrameCallback(handle);
    };
  }, [demoMode, demoStatus]);

  // Show label on new event, then fade after a timeout (not per-frame)
  useEffect(() => {
    if (!latestLabel) return undefined;
    setLabelVisible(true);
    const timer = setTimeout(() => setLabelVisible(false), LABEL_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [latestLabel?.timestamp]);

  const dismissLabel = useCallback(() => setLabelVisible(false), []);

  return (
    <div className="relative w-full h-full rounded-2xl overflow-hidden glass-panel">
      {demoMode ? (
        <video
          key={`${selectedClipId}-${replayTick}`}
          ref={demoVideoRef}
          src={selectedClip.src}
          className="w-full h-full object-cover"
          autoPlay
          muted
          playsInline
          onEnded={handleDemoEnded}
        />
      ) : (
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          className="w-full h-full object-cover"
        />
      )}
      {/* Offscreen capture surface for demo-clip frames — never rendered visibly. */}
      <canvas ref={demoCanvasRef} className="hidden" />

      <div className="absolute top-4 left-4 flex items-center gap-2 text-xs">
        <span
          className={`w-2 h-2 rounded-full ${
            demoMode
              ? demoStatus === "recognized"
                ? "bg-neon"
                : demoStatus === "no-result"
                ? "bg-amber-400"
                : "bg-neon animate-pulse"
              : recordStatus === "recording" || recordStatus === "processing"
              ? "bg-red-500 animate-pulse"
              : recordStatus === "recognized"
              ? "bg-neon"
              : recordStatus === "uncertain" || recordStatus === "no-result"
              ? "bg-amber-400"
              : connected
              ? "bg-neon"
              : "bg-red-500"
          }`}
        />
        <span className="text-neutral-300">
          {demoMode
            ? demoStatus === "playing"
              ? "Analyzing sign..."
              : demoStatus === "processing"
              ? "Processing..."
              : demoStatus === "recognized"
              ? "Recognized"
              : "No result — try replay"
            : recordStatus === "recording"
            ? `Recording... ${recordSecondsLeft}s`
            : recordStatus === "processing"
            ? "Processing..."
            : recordStatus === "recognized"
            ? "Recognized"
            : recordStatus === "uncertain"
            ? `Uncertain (${Math.round((recordConfidence ?? 0) * 100)}%) — try again`
            : recordStatus === "no-result"
            ? "No result — try again"
            : connected
            ? "Live"
            : "Reconnecting..."}
        </span>
      </div>

      <div className="absolute bottom-4 left-4 flex flex-col items-start gap-2">
        {demoMode && (
          <div className="flex items-center gap-1 flex-wrap max-w-xs">
            {DEMO_CLIPS.map((clip) => (
              <button
                key={clip.id}
                type="button"
                onClick={() => startDemo(clip.id)}
                disabled={demoStatus === "playing" || demoStatus === "processing"}
                className={`px-2 py-1 rounded-md text-[11px] transition-colors disabled:opacity-40 ${
                  clip.id === selectedClipId
                    ? "bg-neon/20 text-neon border border-neon/50"
                    : "glass-panel text-neutral-300 hover:text-neon"
                }`}
              >
                {clip.label}
              </button>
            ))}
          </div>
        )}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => (demoMode ? stopDemo() : startDemo())}
            disabled={!demoMode && (recordStatus === "recording" || recordStatus === "processing")}
            className="px-3 py-1.5 rounded-lg text-xs glass-panel border-neon/40 text-neutral-200 hover:text-neon transition-colors disabled:opacity-40"
          >
            {demoMode ? "Switch to live camera" : "Play demo clip"}
          </button>
          {demoMode && (demoStatus === "recognized" || demoStatus === "no-result") && (
            <button
              type="button"
              onClick={() => startDemo()}
              className="px-3 py-1.5 rounded-lg text-xs glass-panel border-neon/40 text-neutral-200 hover:text-neon transition-colors"
            >
              Replay clip
            </button>
          )}
          {!demoMode && (
            <button
              type="button"
              onClick={startRecording}
              disabled={recordStatus === "recording" || recordStatus === "processing"}
              className="px-3 py-1.5 rounded-lg text-xs glass-panel border-red-400/40 text-neutral-200 hover:text-red-400 transition-colors disabled:opacity-40"
            >
              {recordStatus === "recording"
                ? `Recording ${recordSecondsLeft}s`
                : recordStatus === "processing"
                ? "Processing..."
                : "Record sign (any word)"}
            </button>
          )}
        </div>
      </div>

      <AnimatePresence>
        {labelVisible && latestLabel && (
          <motion.div
            key={latestLabel.timestamp}
            initial={{ opacity: 0, scale: 0.85, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.2 }}
            onClick={dismissLabel}
            className="absolute top-4 right-4 px-4 py-2 rounded-xl glass-panel border-neon/40 shadow-lg shadow-neon/20"
          >
            <div className="text-neon font-semibold text-lg">{latestLabel.label}</div>
            <div className="text-xs text-neutral-400">
              {(latestLabel.confidence * 100).toFixed(0)}% confidence
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {transcript?.text && (
        <div className="absolute bottom-20 left-4 right-4 text-center">
          <span
            className={`inline-block px-3 py-1 rounded-lg text-sm glass-panel ${
              transcript.is_final ? "text-white" : "text-neutral-400 italic"
            }`}
          >
            {transcript.text}
          </span>
        </div>
      )}
    </div>
  );
}
