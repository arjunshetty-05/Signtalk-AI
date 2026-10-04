// App.jsx — SignTalk AI web dashboard
//
// Wires everything together: auth gate, main layout with WebcamView +
// SubtitleBar + LanguageSelector + collapsible side panels for
// ConversationHistory and AnalyticsPanel.

import { useState } from "react";
import { useAuth } from "./context/AuthProvider.jsx";
import { offlineModeRef } from "./context/AuthProvider.jsx";
import { useGestureSocket } from "./hooks/useGestureSocket.js";
import { useConversationSocket } from "./hooks/useConversationSocket.js";
import LoginScreen from "./components/LoginScreen.jsx";
import WebcamView from "./components/WebcamView.jsx";
import SubtitleBar from "./components/SubtitleBar.jsx";
import LanguageSelector from "./components/LanguageSelector.jsx";
import ConversationHistory from "./components/ConversationHistory.jsx";
import AnalyticsPanel from "./components/AnalyticsPanel.jsx";

export default function App() {
  const { user, token, loading, logout } = useAuth();
  const [offlineMode, setOfflineMode] = useState(false);
  const { latestLabel, latestSentence, connected, sendFrame } = useGestureSocket(token, offlineMode);
  const { lastEventAt: lastConversationEventAt } = useConversationSocket(token);

  const [displayedSentence, setDisplayedSentence] = useState(null); // {sentence, source, low_confidence, receivedAt}
  const [panelsOpen, setPanelsOpen] = useState(true);

  const toggleOfflineMode = () => {
    const next = !offlineMode;
    offlineModeRef.current = next; // read by the apiClient interceptor (AuthProvider.jsx) — outside React's render tree
    setOfflineMode(next); // triggers re-render + useGestureSocket's reconnect-on-change
  };

  // Keep the displayed (possibly translated) sentence in sync with new events
  const hasTranslation = displayedSentence?.sourceKey === latestSentence?.receivedAt;
  const activeSentenceEvent = latestSentence
    ? {
        ...latestSentence,
        sentence: hasTranslation ? displayedSentence.sentence : latestSentence.sentence,
        lang: hasTranslation ? displayedSentence.lang : "en",
      }
    : null;

  const handleTranslated = (translatedText, lang) => {
    if (!latestSentence) return;
    setDisplayedSentence({
      sourceKey: latestSentence.receivedAt,
      sentence: translatedText,
      lang,
    });
  };

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center text-neutral-400">Loading...</div>;
  }

  if (!user) {
    return <LoginScreen />;
  }

  return (
    <div className="h-screen flex flex-col md:flex-row gap-4 p-4 overflow-hidden">
      <header className="md:hidden flex items-center justify-between">
        <h1 className="text-lg font-semibold text-neon">SignTalk AI</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={toggleOfflineMode}
            className={`text-xs px-2 py-1 rounded-md border ${
              offlineMode ? "border-amber-400 text-amber-400" : "border-neutral-700 text-neutral-400"
            }`}
          >
            {offlineMode ? "✈️" : "Offline"}
          </button>
          <button onClick={() => setPanelsOpen((v) => !v)} className="text-xs text-neutral-400">
            {panelsOpen ? "Hide panels" : "Show panels"}
          </button>
        </div>
      </header>

      <main className="flex-1 flex flex-col gap-4 min-h-0">
        <div className="hidden md:flex items-center justify-between">
          <h1 className="text-xl font-semibold text-neon">SignTalk AI</h1>
          <div className="flex items-center gap-3">
            <button
              onClick={toggleOfflineMode}
              title="Toggle offline (airplane) mode — skips Gemini/network translate, uses Flan-T5 + Coqui only"
              className={`text-xs px-2 py-1 rounded-md border transition-colors ${
                offlineMode
                  ? "border-amber-400 text-amber-400"
                  : "border-neutral-700 text-neutral-400 hover:text-neutral-200"
              }`}
            >
              {offlineMode ? "✈️ Offline mode" : "Offline mode"}
            </button>
            <LanguageSelector currentText={latestSentence?.sentence} onTranslated={handleTranslated} />
            <button onClick={logout} className="text-xs text-neutral-400 hover:text-neutral-200">
              Log out
            </button>
          </div>
        </div>

        <div className="flex-1 relative min-h-0">
          <WebcamView latestLabel={latestLabel} connected={connected} sendFrame={sendFrame} />
        </div>

        <div className="md:hidden">
          <LanguageSelector currentText={latestSentence?.sentence} onTranslated={handleTranslated} />
        </div>
      </main>

      {panelsOpen && (
        <aside className="w-full md:w-80 flex flex-col gap-4 min-h-0 overflow-y-auto">
          <AnalyticsPanel latestLabel={latestLabel} lastConversationEventAt={lastConversationEventAt} />
          <ConversationHistory lastConversationEventAt={lastConversationEventAt} />
        </aside>
      )}

      <SubtitleBar sentenceEvent={activeSentenceEvent} />
    </div>
  );
}
