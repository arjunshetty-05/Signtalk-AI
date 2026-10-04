// ConversationHistory.jsx — SignTalk AI web dashboard
//
// Fetches GET /conversations/{user_id}, renders in reverse-chronological
// order (already sorted server-side), each entry showing sentence,
// language, emotion tag, and timestamp.

import { useEffect, useState } from "react";
import { apiClient } from "../context/AuthProvider.jsx";
import { useAuth } from "../context/AuthProvider.jsx";

const EMOTION_COLORS = {
  happy: "bg-yellow-500/20 text-yellow-300",
  sad: "bg-blue-500/20 text-blue-300",
  angry: "bg-red-500/20 text-red-300",
  fear: "bg-purple-500/20 text-purple-300",
  surprise: "bg-pink-500/20 text-pink-300",
  neutral: "bg-neutral-500/20 text-neutral-300",
  disgust: "bg-green-500/20 text-green-300",
};

export default function ConversationHistory({ lastConversationEventAt }) {
  const { user } = useAuth();
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Re-fetches on mount, and again whenever the Socket.IO "conversation:new"
  // event fires (see useConversationSocket) — so a newly saved conversation
  // shows up live instead of only after a manual page reload.
  useEffect(() => {
    if (!user) return;
    let cancelled = false;

    (async () => {
      setLoading(true);
      setError(null);
      try {
        const { data } = await apiClient.get(`/conversations/${user.uid}`);
        if (!cancelled) setEntries(data);
      } catch (err) {
        if (!cancelled) setError("Could not load conversation history.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [user, lastConversationEventAt]);

  return (
    <div className="glass-panel rounded-2xl p-4 h-full overflow-y-auto">
      <h2 className="text-sm font-semibold text-neutral-300 mb-3">Conversation History</h2>

      {loading && <p className="text-xs text-neutral-500">Loading...</p>}
      {error && <p className="text-xs text-red-400">{error}</p>}
      {!loading && !error && entries.length === 0 && (
        <p className="text-xs text-neutral-500">No conversations yet.</p>
      )}

      <ul className="space-y-2">
        {entries.map((entry, idx) => (
          <li key={`${entry.created_at}-${idx}`} className="rounded-lg bg-white/5 p-3">
            <p className="text-sm text-white">{entry.sentence}</p>
            <div className="flex items-center gap-2 mt-1.5">
              <span className="text-[10px] uppercase tracking-wide text-neutral-500">
                {entry.language}
              </span>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full ${
                  EMOTION_COLORS[entry.emotion] || EMOTION_COLORS.neutral
                }`}
              >
                {entry.emotion}
              </span>
              <span className="text-[10px] text-neutral-600 ml-auto">
                {new Date(entry.created_at * 1000).toLocaleTimeString()}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
