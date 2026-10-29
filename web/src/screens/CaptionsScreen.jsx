import { useEffect, useRef, useState } from "react";
import { useSpeechSocket } from "../hooks/useSpeechSocket.js";

// Direction B UI (Section 2.4): the hearing person speaks, live captions appear
// and build a conversation log. Uses the server Whisper path over /ws/speech;
// if STT is unavailable server-side, the first message says so (graceful
// degradation — the browser's own dictation is the fallback, Section 5.12).

const LANGS = [
  { code: "en", label: "English" },
  { code: "hi", label: "हिन्दी" },
  { code: "kn", label: "ಕನ್ನಡ" },
];

export default function CaptionsScreen() {
  const [lang, setLang] = useState("en");
  const [copyMessage, setCopyMessage] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef(null);
  const {
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
    disconnect,
    setLog,
  } = useSpeechSocket();
  const selectedLanguage = LANGS.find((item) => item.code === lang);
  const active = connected || starting;

  useEffect(() => {
    if (connected && !startedAt.current) startedAt.current = Date.now();
    if (!connected) {
      startedAt.current = null;
      setElapsed(0);
      return undefined;
    }
    const timer = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt.current) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [connected]);

  const formatTime = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  const formatTurnTime = (value) => value
    ? new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "";

  async function copyText(text, message) {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopyMessage(message);
    } catch {
      setCopyMessage("Clipboard access is unavailable in this browser.");
    }
    window.setTimeout(() => setCopyMessage(""), 2500);
  }

  function downloadTranscript() {
    if (!log.length) return;
    const contents = log.map((entry) => {
      const time = formatTurnTime(entry.time);
      const language = LANGS.find((item) => item.code === entry.lang)?.label || entry.lang;
      return `[${time}] ${language}: ${entry.text}`;
    }).join("\n");
    const url = URL.createObjectURL(new Blob([contents], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `signtalk-transcript-${new Date().toISOString().slice(0, 10)}.txt`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <section className="mx-auto max-w-6xl">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="mb-1 text-xs font-bold uppercase tracking-[0.18em] text-[#8b6a38]">Speech to text</p>
          <h2 className="text-2xl font-bold tracking-tight text-[#1d2c25] sm:text-3xl">Voice captions</h2>
          <p className="mt-1 text-sm text-[#69776e]">Turn spoken conversation into readable, timestamped text.</p>
        </div>
        <div>
          <p className="mb-1 text-xs font-medium text-[#68766e]">Spoken language</p>
          <div className="flex gap-1 rounded-lg bg-[#eeede7] p-1" role="group" aria-label="Spoken language">
            {LANGS.map((l) => (
              <button
                key={l.code}
                type="button"
                disabled={active}
                onClick={() => setLang(l.code)}
                aria-pressed={lang === l.code}
                className={`rounded-md px-3 py-2 text-sm font-semibold disabled:opacity-50 ${
                  lang === l.code ? "bg-white text-[#1c5b46] shadow-sm" : "text-[#68766e] hover:text-[#26362e]"
                }`}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && (
        <p className="mb-4 rounded-xl border border-[#f1c6bc] bg-[#fff1ed] p-3 text-[#9b3c2f]" role="alert">
          {error}
        </p>
      )}

      <div className="mb-4 flex flex-wrap items-center gap-2 rounded-xl border border-[#e6e3db] bg-white p-3">
        {active ? (
          <>
            <button
              type="button"
              disabled={!recording}
              onClick={flush}
              className="rounded-lg bg-[#1c5b46] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#164b39] focus-visible:ring active:scale-95 disabled:opacity-50"
            >
              Finish phrase
            </button>
            <button
              type="button"
              onClick={disconnect}
              className="rounded-lg border border-[#dddcd4] bg-white px-4 py-2.5 text-sm font-semibold text-[#52635a] hover:bg-[#f5f4ef]"
            >
              Stop
            </button>
          </>
        ) : starting ? (
          <button
            type="button"
            disabled
            className="rounded-lg bg-[#1c5b46] px-4 py-2.5 text-sm font-semibold text-white opacity-80"
          >
            Requesting microphone…
          </button>
        ) : (
          <button
            type="button"
            onClick={() => start(lang)}
            className="rounded-lg bg-[#1c5b46] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#164b39] active:scale-95 focus-visible:ring"
          >
            🎤 Start listening
          </button>
        )}
        <span
          className={`ml-auto flex items-center gap-2 text-sm ${recording ? "font-semibold text-[#1c5b46]" : "text-[#78847c]"}`}
          role="status"
          aria-live="polite"
        >
          {starting
            ? "Requesting microphone access…"
            : notice && !recording
              ? notice
              : recording
                ? `Listening · ${formatTime(elapsed)}`
                : transcribing
                  ? `Transcribing · ${formatTime(elapsed)}`
                  : connected
                    ? `Connected · ${formatTime(elapsed)}`
                    : "Ready"}
        </span>
      </div>

      <div className="grid items-start gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <section className="overflow-hidden rounded-2xl border border-[#dbe6dc] bg-white shadow-[0_8px_30px_rgba(36,49,40,0.05)]" aria-label="Latest caption">
          <div className="flex items-center justify-between border-b border-[#edf0ea] px-4 py-3 sm:px-5">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-[#8b6a38]">Latest caption</p>
              <p className="mt-0.5 text-xs text-[#78847c]">{selectedLanguage?.label} · captions update about every 2.5 seconds</p>
            </div>
            <span className={`flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${recording ? "bg-[#edf5ee] text-[#356449]" : transcribing ? "bg-[#fff8e9] text-[#916b2c]" : "bg-[#f2f1ec] text-[#68766e]"}`}>
              <span className={`h-2 w-2 rounded-full ${recording ? "animate-pulse bg-[#639273]" : transcribing ? "animate-pulse bg-[#c08a31]" : "bg-[#aeb4aa]"}`} />
              {recording ? (transcribing ? "Listening · updating" : "Listening") : transcribing ? "Working" : "Standby"}
            </span>
          </div>
          <div className="min-h-40 p-4 sm:min-h-52 sm:p-5">
            {caption ? (
              <>
                <p className="text-xl font-semibold leading-relaxed text-[#23352b] sm:text-2xl" aria-live="polite" lang={lang}>{caption}</p>
                <button type="button" onClick={() => copyText(caption, "Latest caption copied")} className="mt-4 rounded-lg border border-[#d8ddd7] px-3 py-2 text-sm font-semibold text-[#52635a] hover:bg-[#f5f7f3] focus-visible:ring">Copy latest</button>
              </>
            ) : (
              <div className="flex h-full min-h-32 flex-col justify-center">
                <p className="text-lg font-semibold text-[#34453b]">Your caption will appear here</p>
                <p className="mt-1 max-w-md text-sm leading-relaxed text-[#78847c]">Choose a language and start listening. Captions update about every 2.5 seconds, and short segments join into one phrase. Finish a phrase or stop to save it as one turn.</p>
                <div className="mt-4 flex flex-wrap gap-2 text-xs font-medium text-[#52635a]">
                  <span className="rounded-full bg-[#f2f5f2] px-3 py-1.5">1 · Select language</span>
                  <span className="rounded-full bg-[#f2f5f2] px-3 py-1.5">2 · Start listening</span>
                  <span className="rounded-full bg-[#f2f5f2] px-3 py-1.5">3 · Speak clearly</span>
                </div>
              </div>
            )}
          </div>
          {notice && <p className="border-t border-[#edf0ea] bg-[#fffaf0] px-4 py-2 text-sm text-[#916b2c]" role="status">{notice}</p>}
        </section>

        <section className="rounded-2xl border border-[#e6e3db] bg-white p-4 shadow-[0_8px_30px_rgba(36,49,40,0.04)] sm:p-5" aria-label="Conversation transcript">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="font-bold text-[#26362e]">Conversation transcript</h3>
              <p className="mt-0.5 text-xs text-[#78847c]">{log.length} {log.length === 1 ? "turn" : "turns"} in this conversation</p>
            </div>
            {log.length > 0 && (
              <button type="button" onClick={() => setLog([])} className="rounded-lg border border-[#dddcd4] px-3 py-1.5 text-xs font-semibold text-[#52635a] hover:bg-[#f5f4ef]">Clear</button>
            )}
          </div>
          {log.length ? (
            <>
              <ul className="max-h-[26rem] space-y-2 overflow-y-auto pr-1">
                {log.map((entry, i) => (
                  <li key={`${entry.time || "turn"}-${i}`} className="rounded-xl border border-[#edf0ea] bg-[#fbfcfa] p-3" lang={entry.lang}>
                    <div className="mb-1.5 flex items-center justify-between gap-2 text-[11px] font-semibold uppercase tracking-wide text-[#78847c]">
                      <span>{LANGS.find((item) => item.code === entry.lang)?.label || entry.lang}</span>
                      <time dateTime={entry.time}>{formatTurnTime(entry.time)}</time>
                    </div>
                    <p className="leading-relaxed text-[#34453b]">{entry.text}</p>
                  </li>
                ))}
              </ul>
              <div className="mt-3 flex flex-wrap gap-2 border-t border-[#edf0ea] pt-3">
                <button type="button" onClick={() => copyText(log.map((entry) => entry.text).join("\n"), "Transcript copied")} className="rounded-lg bg-[#1c5b46] px-3 py-2 text-xs font-semibold text-white hover:bg-[#164b39]">Copy transcript</button>
                <button type="button" onClick={downloadTranscript} className="rounded-lg border border-[#dddcd4] px-3 py-2 text-xs font-semibold text-[#52635a] hover:bg-[#f5f4ef]">Save as .txt</button>
              </div>
            </>
          ) : (
            <div className="flex min-h-36 flex-col justify-center rounded-xl border border-dashed border-[#d8d9d1] bg-[#fbfaf7] p-4">
              <p className="font-semibold text-[#34453b]">No turns yet</p>
              <p className="mt-1 text-sm leading-relaxed text-[#78847c]">Each finished phrase appears here as one turn with its language and time. Your conversation stays visible while the session is open.</p>
            </div>
          )}
          {copyMessage && <p className="mt-2 text-xs font-medium text-[#356449]" role="status">{copyMessage}</p>}
        </section>
      </div>
    </section>
  );
}
