import { useState } from "react";
import { useSpeechSocket } from "../hooks/useSpeechSocket.js";
import GlassButton from "../components/GlassButton.jsx";

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
      <div className="glass p-5 sm:p-6">
        <div className="mb-5 flex items-center justify-end">
          <div
            className="flex gap-1 rounded-xl border border-white/10 bg-white/5 p-1"
            role="group"
            aria-label="Spoken language"
          >
            {LANGS.map((l) => (
              <button
                key={l.code}
                type="button"
                disabled={recording}
                onClick={() => setLang(l.code)}
                aria-pressed={lang === l.code}
                className={`rounded-lg px-3 py-1 text-sm font-medium transition-colors duration-200 disabled:opacity-50 ${
                  lang === l.code
                    ? "bg-brand-600 text-white"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <p
            className="mb-4 rounded-xl border border-red-400/30 bg-red-500/15 p-3 text-red-100"
            role="alert"
          >
            {error}
          </p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          {!recording ? (
            <GlassButton variant="success" onClick={() => start(lang)}>
              🎤 Start listening
            </GlassButton>
          ) : (
            <>
              <GlassButton variant="brand" onClick={flush}>
                Transcribe now
              </GlassButton>
              <GlassButton variant="clear" onClick={disconnect}>
                Stop
              </GlassButton>
            </>
          )}
          <span
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-medium ${
              connected
                ? recording
                  ? "border-emerald-400/40 bg-emerald-500/15 text-emerald-200"
                  : "border-brand-400/40 bg-brand-500/15 text-brand-100"
                : "border-white/10 bg-white/5 text-slate-400"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                connected
                  ? recording
                    ? "bg-emerald-400 animate-pulse"
                    : "bg-brand-400"
                  : "bg-slate-500"
              }`}
            />
            {connected ? (recording ? "listening…" : "connected") : "idle"}
          </span>
        </div>

        {/* Live caption (large, high-contrast). */}
        <div className="mt-6 min-h-[5rem] rounded-xl border border-white/10 bg-ink-900/50 p-5">
          <p className="text-xs uppercase tracking-widest text-slate-400">
            Live caption
          </p>
          <p
            className="mt-1 text-2xl font-semibold text-white"
            aria-live="polite"
            lang={lang}
          >
            {caption ? (
              caption
            ) : recording ? (
              <span className="inline-flex items-end gap-1 text-brand-300" aria-hidden="true">
                <Bar delay="0ms" />
                <Bar delay="150ms" />
                <Bar delay="300ms" />
                <Bar delay="450ms" />
              </span>
            ) : (
              <span className="text-slate-500">…</span>
            )}
          </p>
        </div>

        {/* Conversation log. */}
        <div className="mt-5">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="flex items-center gap-2 text-base font-semibold">
              <span className="h-4 w-1 rounded-full bg-brand-500" />
              Conversation log
            </h3>
            {log.length > 0 && (
              <button
                type="button"
                onClick={() => setLog([])}
                className="rounded-lg border border-white/10 bg-white/5 px-3 py-1 text-sm text-slate-200 transition-colors hover:bg-white/10"
              >
                Clear
              </button>
            )}
          </div>
          <ul className="space-y-2">
            {log.map((entry, i) => (
              <li
                key={i}
                className="animate-fade-up rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2.5"
                lang={entry.lang}
              >
                {entry.text}
              </li>
            ))}
            {log.length === 0 && (
              <li className="text-sm text-slate-500">
                Spoken turns appear here once transcribed.
              </li>
            )}
          </ul>
        </div>
      </div>
    </section>
  );
}

/** Animated equalizer bar for the "listening" state. */
function Bar({ delay }) {
  return (
    <span
      className="inline-block w-1.5 rounded-full bg-current animate-float"
      style={{ height: "1.4rem", animationDelay: delay, animationDuration: "0.9s" }}
    />
  );
}
