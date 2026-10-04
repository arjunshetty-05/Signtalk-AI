// App.jsx — SignTalk AI web dashboard
//
// Wires everything together: auth gate, main layout with WebcamView +
// SubtitleBar + LanguageSelector + collapsible side panels for
// ConversationHistory and AnalyticsPanel.

import { useState } from "react";
import { useAuth } from "./context/AuthProvider.jsx";
import { useGestureSocket } from "./hooks/useGestureSocket.js";
import LoginScreen from "./components/LoginScreen.jsx";
import WebcamView from "./components/WebcamView.jsx";
import SubtitleBar from "./components/SubtitleBar.jsx";
import LanguageSelector from "./components/LanguageSelector.jsx";
import ConversationHistory from "./components/ConversationHistory.jsx";
import AnalyticsPanel from "./components/AnalyticsPanel.jsx";

export default function App() {
  const { user, token, loading, logout } = useAuth();
  const { latestLabel, latestSentence, connected, sendFrame } = useGestureSocket(token);

  const [displayedSentence, setDisplayedSentence] = useState(null); // {sentence, source, low_confidence, receivedAt}
  const [panelsOpen, setPanelsOpen] = useState(true);

  // Keep the displayed (possibly translated) sentence in sync with new events
  const activeSentenceEvent = latestSentence
    ? { ...latestSentence, sentence: displayedSentence?.sourceKey === latestSentence.receivedAt
        ? displayedSentence.sentence
        : latestSentence.sentence }
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
    <div className="min-h-screen flex flex-col md:flex-row gap-4 p-4">
      <header className="md:hidden flex items-center justify-between">
        <h1 className="text-lg font-semibold text-neon">SignTalk AI</h1>
        <button onClick={() => setPanelsOpen((v) => !v)} className="text-xs text-neutral-400">
          {panelsOpen ? "Hide panels" : "Show panels"}
        </button>
      </header>

      <main className="flex-1 flex flex-col gap-4 min-h-[60vh]">
        <div className="hidden md:flex items-center justify-between">
          <h1 className="text-xl font-semibold text-neon">SignTalk AI</h1>
          <div className="flex items-center gap-3">
            <LanguageSelector currentText={latestSentence?.sentence} onTranslated={handleTranslated} />
            <button onClick={logout} className="text-xs text-neutral-400 hover:text-neutral-200">
              Log out
            </button>
          </div>
        </div>

        <div className="flex-1 relative min-h-[400px]">
          <WebcamView latestLabel={latestLabel} connected={connected} sendFrame={sendFrame} />
        </div>

        <div className="md:hidden">
          <LanguageSelector currentText={latestSentence?.sentence} onTranslated={handleTranslated} />
        </div>
      </main>

      {panelsOpen && (
        <aside className="w-full md:w-80 flex flex-col gap-4">
          <AnalyticsPanel latestLabel={latestLabel} />
          <ConversationHistory />
        </aside>
      )}

      <SubtitleBar sentenceEvent={activeSentenceEvent} />
    </div>
  );
}
