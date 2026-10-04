// useConversationSocket.js — SignTalk AI web dashboard
//
// Subscribes to the backend's Socket.IO "conversation:new" event, so a
// second device (or the same tab) sees newly saved conversations without
// polling. This is a *separate* transport from useGestureSocket — that one
// talks to the plain FastAPI /ws/gesture WebSocket; this one talks to the
// Socket.IO server mounted in api/socket_manager.py.
//
// The server's `connect` handler verifies `auth: {token}` itself and joins
// the socket to a room keyed by the token's own uid — never trust a
// client-supplied uid, so we only ever send the token here.

import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import { API_BASE_URL } from "../firebase.js";

export function useConversationSocket(token) {
  const [lastEventAt, setLastEventAt] = useState(null);
  const socketRef = useRef(null);

  useEffect(() => {
    if (!token) return undefined;

    const socket = io(API_BASE_URL, {
      auth: { token },
      transports: ["websocket"],
    });
    socketRef.current = socket;

    socket.on("conversation:new", () => {
      setLastEventAt(Date.now());
    });

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, [token]);

  return { lastEventAt };
}
