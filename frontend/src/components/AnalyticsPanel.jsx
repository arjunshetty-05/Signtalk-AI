// AnalyticsPanel.jsx — SignTalk AI web dashboard
//
// Two Recharts charts: a line chart of prediction confidence over the last
// N gesture events (a rolling buffer of the sparse, debounced /ws/gesture
// events — NOT per-frame samples), and a bar chart of emotion distribution
// for the session, sourced from /conversations/{user_id} (each conversation
// entry carries an emotion tag).

import { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, CartesianGrid,
} from "recharts";
import { apiClient } from "../context/AuthProvider.jsx";
import { useAuth } from "../context/AuthProvider.jsx";

const MAX_CONFIDENCE_POINTS = 30;

export default function AnalyticsPanel({ latestLabel }) {
  const { user } = useAuth();
  const [confidenceHistory, setConfidenceHistory] = useState([]);
  const [emotionCounts, setEmotionCounts] = useState({});

  // Rolling buffer of sparse gesture-confidence events
  useEffect(() => {
    if (!latestLabel) return;
    setConfidenceHistory((prev) => {
      const next = [...prev, { time: new Date(latestLabel.timestamp * 1000).toLocaleTimeString(), confidence: latestLabel.confidence }];
      return next.slice(-MAX_CONFIDENCE_POINTS);
    });
  }, [latestLabel?.timestamp]);

  // Emotion distribution from conversation history
  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    (async () => {
      try {
        const { data } = await apiClient.get(`/conversations/${user.uid}`);
        if (cancelled) return;
        const counts = {};
        for (const entry of data) {
          counts[entry.emotion] = (counts[entry.emotion] || 0) + 1;
        }
        setEmotionCounts(counts);
      } catch {
        /* analytics is best-effort — fail silently */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user, latestLabel?.timestamp]);

  const emotionData = useMemo(
    () => Object.entries(emotionCounts).map(([emotion, count]) => ({ emotion, count })),
    [emotionCounts]
  );

  return (
    <div className="glass-panel rounded-2xl p-4 space-y-6">
      <div>
        <h2 className="text-sm font-semibold text-neutral-300 mb-2">Confidence over time</h2>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={confidenceHistory}>
            <XAxis dataKey="time" hide />
            <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "#6b7280" }} width={28} />
            <Tooltip contentStyle={{ background: "#111827", border: "none", fontSize: 12 }} />
            <Line type="monotone" dataKey="confidence" stroke="#39ff9d" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h2 className="text-sm font-semibold text-neutral-300 mb-2">Emotion distribution</h2>
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={emotionData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="emotion" tick={{ fontSize: 10, fill: "#6b7280" }} />
            <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: "#6b7280" }} width={20} />
            <Tooltip contentStyle={{ background: "#111827", border: "none", fontSize: 12 }} />
            <Bar dataKey="count" fill="#39ff9d" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
