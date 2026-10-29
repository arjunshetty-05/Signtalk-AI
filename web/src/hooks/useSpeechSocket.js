import { useCallback, useEffect, useRef, useState } from "react";

// Send short rolling chunks so the first caption doesn't wait for a long
// recording window. Adjacent results are still joined into one phrase.
const CAPTION_SEGMENT_MS = 2500;

// Direction B client (Section 6.4): capture mic audio with MediaRecorder,
// stream chunks over WS /ws/speech, and combine short transcript segments into
// one live caption and one conversation-log turn. The browser Web Speech API is a
// possible fallback but this hook uses the server Whisper path (Section 5.12).

// Dev: Vite proxies /api but not /ws by default, so target the server origin
// for the socket. In production the same origin serves both.
function wsUrl(path) {
  const loc = window.location;
  const proto = loc.protocol === "https:" ? "wss:" : "ws:";
  // In Vite dev (port 5173) the FastAPI server is on :8000.
  const host = loc.port === "5173" ? `${loc.hostname}:8000` : loc.host;
  return `${proto}//${host}${path}`;
}

export function useSpeechSocket() {
  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);
  const [starting, setStarting] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [caption, setCaption] = useState(""); // latest transcript
  const [log, setLog] = useState([]); // [{text, lang, time}]
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const wsRef = useRef(null);
  const recorderRef = useRef(null);
  const startRecorderRef = useRef(null);
  const segmentTimerRef = useRef(null);
  const streamRef = useRef(null);
  const sessionActiveRef = useRef(false);
  const pendingTranscriptionsRef = useRef(0);
  const audioQueueRef = useRef(Promise.resolve());
  const turnTextRef = useRef("");
  const turnStartedAtRef = useRef(null);
  const phraseBoundaryRef = useRef(false);
  const finishRequestedRef = useRef(false);
  const closeAfterTurnRef = useRef(false);
  const completeTurnRef = useRef(null);

  const disconnect = useCallback(() => {
    sessionActiveRef.current = false;
    pendingTranscriptionsRef.current = 0;
    clearTimeout(segmentTimerRef.current);
    segmentTimerRef.current = null;
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.onstop = null;
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        /* ignore */
      }
      wsRef.current = null;
    }
    setRecording(false);
    setConnected(false);
    setStarting(false);
    setTranscribing(false);
  }, []);

  const startRecorder = useCallback((stream, ws) => {
    if (!sessionActiveRef.current || ws.readyState !== WebSocket.OPEN) return;

    try {
      const options = MediaRecorder.isTypeSupported("audio/webm")
        ? { mimeType: "audio/webm" }
        : undefined;
      if (!turnStartedAtRef.current) turnStartedAtRef.current = new Date();
      const rec = new MediaRecorder(stream, options);
      recorderRef.current = rec;
      rec.ondataavailable = (event) => {
        if (event.data.size === 0) return;
        audioQueueRef.current = audioQueueRef.current.then(async () => {
          const audio = await event.data.arrayBuffer();
          if (ws.readyState === WebSocket.OPEN) ws.send(audio);
        });
      };
      rec.onstop = async () => {
        if (!sessionActiveRef.current || ws.readyState !== WebSocket.OPEN) return;

        pendingTranscriptionsRef.current += 1;
        setTranscribing(true);
        audioQueueRef.current = audioQueueRef.current
          .then(() => {
            if (ws.readyState === WebSocket.OPEN) ws.send("flush");
          })
          .catch((err) => {
            setError(err?.message || "Could not send the recording for transcription.");
            pendingTranscriptionsRef.current = Math.max(
              0,
              pendingTranscriptionsRef.current - 1
            );
            setTranscribing(pendingTranscriptionsRef.current > 0);
            audioQueueRef.current = Promise.resolve();
          });

        // Automatic five-second chunks belong to one spoken turn. Only start a
        // fresh chunk when the user has not ended the phrase/session.
        if (!phraseBoundaryRef.current) startRecorderRef.current?.(stream, ws);
      };
      rec.start();
      setRecording(true);
      clearTimeout(segmentTimerRef.current);
      segmentTimerRef.current = setTimeout(() => {
        if (recorderRef.current !== rec || rec.state !== "recording") return;
        setRecording(false);
        rec.stop();
      }, CAPTION_SEGMENT_MS);
    } catch (err) {
      setError(err?.message || "Could not start audio recording.");
      disconnect();
    }
  }, [disconnect]);
  startRecorderRef.current = startRecorder;

  const start = useCallback(
    async (lang = "en") => {
      setError(null);
      setNotice(null);
      setCaption("");
      turnTextRef.current = "";
      turnStartedAtRef.current = null;
      phraseBoundaryRef.current = false;
      closeAfterTurnRef.current = false;
      setStarting(true);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;
        sessionActiveRef.current = true;

        const ws = new WebSocket(wsUrl("/ws/speech"));
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        const completeTurn = () => {
          const text = turnTextRef.current.trim();
          if (text) {
            setLog((prev) => [
              ...prev,
              {
                text,
                lang,
                time: (turnStartedAtRef.current || new Date()).toISOString(),
              },
            ]);
          }
          turnTextRef.current = "";
          turnStartedAtRef.current = null;
          phraseBoundaryRef.current = false;
          finishRequestedRef.current = false;
          if (closeAfterTurnRef.current) {
            closeAfterTurnRef.current = false;
            disconnect();
          } else if (sessionActiveRef.current && ws.readyState === WebSocket.OPEN) {
            startRecorder(stream, ws);
          }
        };
        completeTurnRef.current = completeTurn;

        ws.onopen = () => {
          setConnected(true);
          setStarting(false);
          ws.send(`lang:${lang}`);
          startRecorder(stream, ws);
        };

        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === "transcript") {
              setError(null);
              setNotice(null);
              if (msg.text) {
                const segment = msg.text.trim();
                turnTextRef.current = [turnTextRef.current.trim(), segment]
                  .filter(Boolean)
                  .join(" ");
                setCaption(turnTextRef.current);
              }
            } else if (msg.type === "no_speech") {
              setNotice("No speech detected in the latest segment; keep speaking.");
            } else if (msg.type === "error") {
              setNotice(null);
              setError(msg.text || "Speech transcription failed.");
            }
            if (pendingTranscriptionsRef.current > 0) {
              pendingTranscriptionsRef.current -= 1;
              setTranscribing(pendingTranscriptionsRef.current > 0);
            }
            if (phraseBoundaryRef.current && pendingTranscriptionsRef.current === 0) {
              completeTurnRef.current?.();
            }
          } catch {
            /* ignore non-JSON */
          }
        };

        ws.onerror = () => {
          setError("Speech connection error. Check that the backend is running.");
          disconnect();
        };
        ws.onclose = () => {
          sessionActiveRef.current = false;
          setConnected(false);
          setRecording(false);
          setStarting(false);
          pendingTranscriptionsRef.current = 0;
          setTranscribing(false);
          clearTimeout(segmentTimerRef.current);
          segmentTimerRef.current = null;
          if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
          }
          recorderRef.current = null;
        };
      } catch (err) {
        setError(err?.message || "Could not access the microphone.");
        disconnect();
      }
    },
    [disconnect, startRecorder]
  );

  // Close a spoken phrase after its current segment and all queued segments
  // have been transcribed. By default listening resumes for the next phrase.
  const finishTurn = useCallback(({ stopListening = false } = {}) => {
    phraseBoundaryRef.current = true;
    finishRequestedRef.current = true;
    closeAfterTurnRef.current = stopListening;
    const rec = recorderRef.current;
    clearTimeout(segmentTimerRef.current);
    segmentTimerRef.current = null;
    if (rec?.state === "recording" && wsRef.current?.readyState === WebSocket.OPEN) {
      setError(null);
      setRecording(false);
      rec.stop();
    } else if (pendingTranscriptionsRef.current === 0) {
      completeTurnRef.current?.();
    }
  }, []);

  // Preserve and commit the final partial chunk before closing the socket.
  const flush = useCallback(() => finishTurn({ stopListening: false }), [finishTurn]);
  const stop = useCallback(() => {
    clearTimeout(segmentTimerRef.current);
    segmentTimerRef.current = null;

    // A submitted Finish phrase is the only work Stop should wait for. Stop
    // the microphone now, then let that phrase's pending transcription commit.
    if (
      finishRequestedRef.current &&
      phraseBoundaryRef.current &&
      wsRef.current?.readyState === WebSocket.OPEN
    ) {
      sessionActiveRef.current = false;
      closeAfterTurnRef.current = true;
      setRecording(false);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      return;
    }

    // Otherwise discard any unfinished audio and close the socket immediately.
    disconnect();
  }, [disconnect]);

  useEffect(() => disconnect, [disconnect]);

  return {
    connected,
    recording,
    starting,
    transcribing,
    caption,
    log,
    error,
    notice,
    start,
    flush,
    disconnect: stop,
    setLog,
  };
}
