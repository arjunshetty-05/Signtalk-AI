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

import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { motion, AnimatePresence } from "framer-motion";
import { WS_BASE_URL } from "../firebase.js";
import { useAuth } from "../context/AuthProvider.jsx";

const CAPTURE_FPS = 20;
const LABEL_TIMEOUT_MS = 4000;

function useSpeechSocket(token) {
  const [transcript, setTranscript] = useState(null); // {text, is_final}
  const socketRef = useRef(null);

  useEffect(() => {
    if (!token) return undefined;
    const socket = new WebSocket(`${WS_BASE_URL}/ws/speech?token=${encodeURIComponent(token)}`);
    socketRef.current = socket;
    socket.onmessage = (event) => {
      try {
        setTranscript(JSON.parse(event.data));
      } catch {
        /* ignore malformed frame */
      }
    };
    return () => socket.close();
  }, [token]);

  return transcript;
}

export default function WebcamView({ latestLabel, connected, sendFrame }) {
  const { token } = useAuth();
  const webcamRef = useRef(null);
  const transcript = useSpeechSocket(token);
  const [labelVisible, setLabelVisible] = useState(false);

  // Stream frames at ~CAPTURE_FPS
  useEffect(() => {
    const interval = setInterval(() => {
      const screenshot = webcamRef.current?.getScreenshot();
      if (screenshot) sendFrame(screenshot);
    }, 1000 / CAPTURE_FPS);
    return () => clearInterval(interval);
  }, [sendFrame]);

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
      <Webcam
        ref={webcamRef}
        audio={false}
        mirrored
        screenshotFormat="image/jpeg"
        className="w-full h-full object-cover"
      />

      <div className="absolute top-4 left-4 flex items-center gap-2 text-xs">
        <span className={`w-2 h-2 rounded-full ${connected ? "bg-neon" : "bg-red-500"}`} />
        <span className="text-neutral-300">{connected ? "Live" : "Reconnecting..."}</span>
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
