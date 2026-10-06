import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Open the webcam via getUserMedia and expose a ready-to-attach <video> ref
 * plus a start/stop frame grabber.
 *
 * The preview element may be CSS-mirrored for comfort, but the pixels captured
 * here are drawn straight from the raw stream with no horizontal flip, so the
 * frames uploaded to the server are never mirrored (D1 companion rule in
 * docs/DECISIONS.md).
 *
 * @param {{width?: number, height?: number}} [opts]
 */
export function useWebcam(opts = {}) {
  const { width = 640, height = 480 } = opts;
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const grabTimerRef = useRef(null);
  const framesRef = useRef([]);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function open() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width, height, facingMode: "user" },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setReady(true);
      } catch (err) {
        if (!cancelled) {
          setError(err?.message || "Could not access the webcam.");
        }
      }
    }

    open();
    return () => {
      cancelled = true;
      if (grabTimerRef.current) clearInterval(grabTimerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, [width, height]);

  /**
   * Begin grabbing UN-mirrored base64 JPEG frames at `fps`.
   *
   * @param {number} fps  target capture frame rate
   * @param {(count: number) => void} [onFrame]  per-frame progress callback
   */
  const startCapture = useCallback(
    (fps, onFrame) => {
      if (grabTimerRef.current) return; // already recording
      const video = videoRef.current;
      if (!video) return;

      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || width;
      canvas.height = video.videoHeight || height;
      const ctx = canvas.getContext("2d");

      framesRef.current = [];

      const grab = () => {
        // Draw the raw (un-mirrored) frame; no scaleX flip here.
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
        framesRef.current.push(dataUrl.split(",", 2)[1]); // strip data: prefix
        if (onFrame) onFrame(framesRef.current.length);
      };

      grab();
      grabTimerRef.current = setInterval(grab, 1000 / fps);
    },
    [width, height]
  );

  /**
   * Stop grabbing and return the frames collected since `startCapture`.
   * @returns {string[]} the captured base64 JPEG frames
   */
  const stopCapture = useCallback(() => {
    if (grabTimerRef.current) {
      clearInterval(grabTimerRef.current);
      grabTimerRef.current = null;
    }
    const frames = framesRef.current;
    framesRef.current = [];
    return frames;
  }, []);

  return { videoRef, ready, error, startCapture, stopCapture };
}
