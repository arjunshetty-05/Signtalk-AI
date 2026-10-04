// useGestureSocket.js — SignTalk AI web dashboard
//
// Encapsulates the /ws/gesture WebSocket connection lifecycle (connect,
// reconnect, disconnect) in one hook, isolated from any component, so the
// URL/connection logic is easy to swap once the backend is fully live.
//
// NOTE: the backend's /ws/gesture is a plain FastAPI WebSocket (not
// Socket.IO) — auth is passed as a `?token=<firebase JWT>` query param,
// since browsers can't set custom headers on a WebSocket handshake.
//
// Events are sparse/debounced (not one-per-frame), so this hook exposes a
// `receivedAt` timestamp alongside each piece of state — consumers decide
// when to fade a stale label out, rather than assuming a new value every
// render.

import { useCallback, useEffect, useRef, useState } from "react";
import { WS_BASE_URL } from "../firebase.js";

const RECONNECT_DELAY_MS = 2000;

export function useGestureSocket(token) {
  const [latestLabel, setLatestLabel] = useState(null); // {label, confidence, timestamp, receivedAt}
  const [latestSentence, setLatestSentence] = useState(null); // {sentence, source, low_confidence, receivedAt}
  const [connected, setConnected] = useState(false);

  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(true);

  const connect = useCallback(() => {
    if (!token) return;

    const url = `${WS_BASE_URL}/ws/gesture?token=${encodeURIComponent(token)}`;
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = () => setConnected(true);

    socket.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      if (data.error) return;

      if (data.type === "corrected_sentence") {
        setLatestSentence({ ...data, receivedAt: Date.now() });
      } else if (data.label) {
        setLatestLabel({ ...data, receivedAt: Date.now() });
      }
    };

    socket.onclose = () => {
      setConnected(false);
      if (shouldReconnectRef.current) {
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
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

  const sendFrame = useCallback((base64Jpeg) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ frame: base64Jpeg }));
    }
  }, []);

  return { latestLabel, latestSentence, connected, sendFrame };
}
