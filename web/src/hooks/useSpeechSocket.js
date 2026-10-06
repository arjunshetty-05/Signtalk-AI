import { useCallback, useEffect, useRef, useState } from "react";

// Direction B client (Section 6.4): capture mic audio with MediaRecorder,
// stream chunks over WS /ws/speech, and collect {type:"transcript"} messages
// into a live caption + a conversation log. The browser Web Speech API is a
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
  const [caption, setCaption] = useState(""); // latest transcript
  const [log, setLog] = useState([]); // [{text, lang}]
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);

  const disconnect = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
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
  }, []);

  const start = useCallback(
    async (lang = "en") => {
      setError(null);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;

        const ws = new WebSocket(wsUrl("/ws/speech"));
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          ws.send(`lang:${lang}`);
          const rec = new MediaRecorder(stream, { mimeType: "audio/webm" });
          recorderRef.current = rec;
          rec.ondataavailable = (e) => {
            if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
              e.data.arrayBuffer().then((buf) => ws.send(buf));
            }
          };
          rec.start(1000); // emit a chunk each second
          setRecording(true);
        };

        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === "transcript") {
              setCaption(msg.text);
              if (msg.is_final && msg.text) {
                setLog((prev) => [...prev, { text: msg.text, lang: msg.lang }]);
              }
            }
          } catch {
            /* ignore non-JSON */
          }
        };

        ws.onerror = () => setError("Speech connection error.");
        ws.onclose = () => {
          setConnected(false);
          setRecording(false);
        };
      } catch (err) {
        setError(err?.message || "Could not access the microphone.");
        disconnect();
      }
    },
    [disconnect]
  );

  // Ask the server to transcribe the audio buffered so far.
  const flush = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send("flush");
    }
  }, []);

  useEffect(() => disconnect, [disconnect]);

  return { connected, recording, caption, log, error, start, flush, disconnect, setLog };
}
