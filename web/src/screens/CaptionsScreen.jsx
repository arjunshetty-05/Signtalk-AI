import { useState } from "react";
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
  const { connected, recording, caption, log, error, start, flush, disconnect, setLog } =
    useSpeechSocket();

  return (
    <section className="mx-auto max-w-xl">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-bold">Voice to captions</h2>
        <div className="flex gap-1" role="group" aria-label="Spoken language">
          {LANGS.map((l) => (
            <button
              key={l.code}
              type="button"
              disabled={recording}
              onClick={() => setLang(l.code)}
              aria-pressed={lang === l.code}
              className={`rounded px-3 py-1 text-sm disabled:opacity-50 ${
                lang === l.code ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-200"
              }`}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <p className="mb-4 rounded bg-red-900 p-3 text-red-100" role="alert">
          {error}
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        {!recording ? (
          <button
            type="button"
            onClick={() => start(lang)}
            className="rounded-lg bg-green-600 px-4 py-3 font-semibold text-white hover:bg-green-500 focus-visible:ring"
          >
            🎤 Start listening
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={flush}
              className="rounded-lg bg-blue-600 px-4 py-3 font-semibold text-white hover:bg-blue-500 focus-visible:ring"
            >
              Transcribe now
            </button>
            <button
              type="button"
              onClick={disconnect}
              className="rounded-lg bg-gray-700 px-4 py-3 font-semibold text-white hover:bg-gray-600"
            >
              Stop
            </button>
          </>
        )}
        <span className="self-center text-sm text-gray-400">
          {connected ? (recording ? "listening…" : "connected") : "idle"}
        </span>
      </div>

      {/* Live caption (large, high-contrast). */}
      <div className="mt-6 min-h-[4rem] rounded-lg border border-gray-700 p-4">
        <p className="text-sm text-gray-400">Live caption</p>
        <p className="text-2xl font-semibold text-white" aria-live="polite" lang={lang}>
          {caption || "…"}
        </p>
      </div>

      {/* Conversation log. */}
      <div className="mt-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="font-bold">Conversation log</h3>
          {log.length > 0 && (
            <button
              type="button"
              onClick={() => setLog([])}
              className="rounded bg-gray-700 px-3 py-1 text-sm hover:bg-gray-600"
            >
              Clear
            </button>
          )}
        </div>
        <ul className="space-y-2">
          {log.map((entry, i) => (
            <li key={i} className="rounded bg-gray-800 px-3 py-2" lang={entry.lang}>
              {entry.text}
            </li>
          ))}
          {log.length === 0 && (
            <li className="text-sm text-gray-500">
              Spoken turns appear here once transcribed.
            </li>
          )}
        </ul>
      </div>
    </section>
  );
}
