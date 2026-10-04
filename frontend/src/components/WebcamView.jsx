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

export default function WebcamView({ latestLabel, connected, sendFrame, onDemoResult }) {
  const { token } = useAuth();
  const webcamRef = useRef(null);
  const demoVideoRef = useRef(null);
  const demoCanvasRef = useRef(null);
  const demoFramesRef = useRef([]); // collected base64 JPEGs for the current playthrough
  const transcript = useSpeechSocket(token);
  const [labelVisible, setLabelVisible] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
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

  const startDemo = useCallback((clipId) => {
    if (clipId) setSelectedClipId(clipId);
    setDemoMode(true);
    setDemoStatus("playing");
    demoFramesRef.current = [];
    setReplayTick((t) => t + 1); // forces the <video> to remount and play from frame 0
  }, []);

  const stopDemo = useCallback(() => {
    setDemoMode(false);
    setDemoStatus("idle");
  }, []);

  const handleDemoEnded = useCallback(async () => {
    setDemoStatus("processing");
    const frames = demoFramesRef.current;
    if (frames.length < 2) {
      setDemoStatus("no-result");
      return;
    }
    try {
      const { data } = await apiClient.post("/pose/classify-clip", { frames });
      onDemoResult?.(data);
      setDemoStatus("recognized");
    } catch {
      setDemoStatus("no-result");
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
            className="px-3 py-1.5 rounded-lg text-xs glass-panel border-neon/40 text-neutral-200 hover:text-neon transition-colors"
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
