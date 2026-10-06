import { useEffect, useRef, useState } from "react";

// Pre-flight thresholds (L2). These drive only the on-screen GREEN/AMBER
// guidance in the browser — they are NOT the accuracy-critical gate. The server
// re-extracts landmarks and runs the authoritative quality gate (Section 5.3).
const MIN_BRIGHTNESS = 0.28; // mean luma in [0,1]; below this the room is dark
const MAX_BRIGHTNESS = 0.92; // above this the frame is blown out
const MIN_MOTION_AREA = 0.015; // fraction of pixels changing frame-to-frame
const SAMPLE_W = 64; // downsampled analysis canvas (cheap)
const SAMPLE_H = 48;
const INTERVAL_MS = 300;

/**
 * Lightweight pre-flight check (L2). Samples the raw webcam into a tiny canvas
 * and reports brightness + whether there is some subject motion, so the signer
 * can fix framing/lighting BEFORE recording. Calls `onReadyChange(bool)` so the
 * parent can gate recording until the lights are green.
 *
 * This intentionally avoids loading MediaPipe in the browser: it is live
 * guidance, not the recogniser's quality gate.
 *
 * @param {{videoRef: React.RefObject<HTMLVideoElement>, ready: boolean,
 *          onReadyChange?: (ready: boolean) => void}} props
 */
export default function PreflightCheck({ videoRef, ready, onReadyChange }) {
  const canvasRef = useRef(null);
  const prevRef = useRef(null);
  const [brightness, setBrightness] = useState(0);
  const [motion, setMotion] = useState(0);

  useEffect(() => {
    if (!ready) return undefined;
    const canvas = canvasRef.current || document.createElement("canvas");
    canvas.width = SAMPLE_W;
    canvas.height = SAMPLE_H;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    const timer = setInterval(() => {
      const video = videoRef.current;
      if (!video || !video.videoWidth) return;
      ctx.drawImage(video, 0, 0, SAMPLE_W, SAMPLE_H);
      const { data } = ctx.getImageData(0, 0, SAMPLE_W, SAMPLE_H);

      // Mean luma + frame-to-frame change (cheap motion proxy).
      let sum = 0;
      let changed = 0;
      const prev = prevRef.current;
      const luma = new Float32Array(SAMPLE_W * SAMPLE_H);
      for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
        const y = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 255;
        luma[p] = y;
        sum += y;
        if (prev && Math.abs(y - prev[p]) > 0.12) changed += 1;
      }
      prevRef.current = luma;
      setBrightness(sum / (SAMPLE_W * SAMPLE_H));
      setMotion(prev ? changed / (SAMPLE_W * SAMPLE_H) : 0);
    }, INTERVAL_MS);

    return () => clearInterval(timer);
  }, [ready, videoRef]);

  const brightnessOk = brightness >= MIN_BRIGHTNESS && brightness <= MAX_BRIGHTNESS;
  // Motion is informational — not required to be "green" to record (a still
  // start is normal), so framing readiness = camera ready + decent light.
  const allOk = ready && brightnessOk;

  useEffect(() => {
    if (onReadyChange) onReadyChange(allOk);
  }, [allOk, onReadyChange]);

  return (
    <div
      className="mt-3 grid grid-cols-3 gap-2 text-sm"
      role="group"
      aria-label="Pre-flight checks"
    >
      <Light ok={ready} label="Camera" />
      <Light
        ok={brightnessOk}
        label={
          !brightnessOk && brightness < MIN_BRIGHTNESS
            ? "Too dark"
            : !brightnessOk
              ? "Too bright"
              : "Lighting"
        }
      />
      <Light ok={motion >= MIN_MOTION_AREA} label="Motion" soft />
      <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
    </div>
  );
}

/** A single green/amber pre-flight indicator. `soft` = advisory, not blocking. */
function Light({ ok, label, soft = false }) {
  const color = ok
    ? "bg-green-600 text-green-50"
    : soft
      ? "bg-gray-700 text-gray-300"
      : "bg-amber-700 text-amber-50";
  return (
    <div className={`flex items-center gap-2 rounded px-2 py-1 ${color}`}>
      <span aria-hidden="true">{ok ? "●" : "○"}</span>
      <span>{label}</span>
    </div>
  );
}
