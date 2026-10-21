import { useCallback, useEffect, useRef, useState } from "react";
import { composeSentence } from "../api.js";
import { useSpeech } from "../hooks/useSpeech.js";
import GlassButton from "./GlassButton.jsx";

// Sentence buffer (PROJECT_CONTEXT 5.10): collect accepted/confirmed words,
// then compose + speak. Flush triggers: idle gap, max words, or "Finish".
const IDLE_MS = 2500;
const MAX_WORDS = 8;
const APP_LANGS = [
  { code: "en", label: "English" },
  { code: "hi", label: "हिन्दी" },
  { code: "kn", label: "ಕನ್ನಡ" },
];

/**
 * Collects signed words into a sentence, composes en/hi/kn via /api/compose,
 * and speaks the chosen language with the browser's voice.
 *
 * Strict mode (default, demo-safe, Section 5.10): the sentence is shown and
 * spoken only after the user taps "Speak". Auto-compose happens on flush so the
 * text is ready; speaking is the explicit tap.
 *
 * @param {{words: string[], onClear: () => void}} props
 *   `words` is the running list of accepted/confirmed words (owned by parent).
 */
export default function SentenceBox({ words, onClear }) {
  const [lang, setLang] = useState("en");
  const [composed, setComposed] = useState(null); // {sentences, source, verified}
  const [composing, setComposing] = useState(false);
  const idleTimer = useRef(null);
  const { speak, speaking, cancel, supported } = useSpeech();

  const doCompose = useCallback(async () => {
    if (!words.length) return;
    setComposing(true);
    try {
      const data = await composeSentence({ words });
      setComposed(data);
    } catch {
      setComposed(null);
    } finally {
      setComposing(false);
    }
  }, [words]);

  // Flush on idle gap or max words: compose (but don't auto-speak in strict mode).
  useEffect(() => {
    if (idleTimer.current) clearTimeout(idleTimer.current);
    if (!words.length) {
      setComposed(null);
      return undefined;
    }
    if (words.length >= MAX_WORDS) {
      doCompose();
      return undefined;
    }
    idleTimer.current = setTimeout(doCompose, IDLE_MS);
    return () => idleTimer.current && clearTimeout(idleTimer.current);
  }, [words, doCompose]);

  const sentence = composed?.sentences?.[lang] || "";

  return (
    <section
      className="mt-6 rounded-xl border border-white/8 bg-white/[0.02] p-5"
      aria-label="Sentence"
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-base font-semibold">
          <span className="h-4 w-1 rounded-full bg-brand-500" />
          Sentence
        </h3>
        <div
          className="flex gap-1 rounded-xl border border-white/10 bg-white/5 p-1"
          role="group"
          aria-label="Language"
        >
          {APP_LANGS.map((l) => (
            <button
              key={l.code}
              type="button"
              onClick={() => setLang(l.code)}
              aria-pressed={lang === l.code}
              className={`rounded-lg px-3 py-1 text-sm font-medium transition-colors duration-200 ${
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

      {/* Running word chips */}
      <div className="mb-4 flex min-h-[2rem] flex-wrap gap-2">
        {words.length ? (
          words.map((w, i) => (
            <span
              key={`${w}-${i}`}
              style={{ animationDelay: `${i * 50}ms` }}
              className="animate-pop-in rounded-full border border-brand-400/30 bg-brand-500/15 px-3 py-1 text-sm font-medium text-brand-100"
            >
              {w}
            </span>
          ))
        ) : (
          <span className="text-sm text-slate-500">
            Accepted words collect here, then become a sentence.
          </span>
        )}
      </div>

      {/* Composed sentence (large, high-contrast subtitle) */}
      <p
        className="min-h-[3rem] text-2xl font-semibold leading-snug text-white"
        aria-live="polite"
        lang={lang}
      >
        {composing ? <span className="shimmer-text">Composing…</span> : sentence}
      </p>
      {composed && (
        <p className="mt-1 text-xs text-slate-400">
          source: {composed.source}
          {composed.verified
            ? " (verified)"
            : " (unverified — review before trusting)"}
        </p>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <GlassButton
          variant="success"
          disabled={!sentence || !supported || speaking}
          onClick={() => speak(sentence, lang)}
        >
          🔊 Speak
        </GlassButton>
        {speaking && (
          <GlassButton variant="clear" onClick={cancel}>
            Stop
          </GlassButton>
        )}
        <GlassButton
          variant="clear"
          disabled={!words.length}
          onClick={() => {
            cancel();
            setComposed(null);
            onClear();
          }}
        >
          Finish / Clear
        </GlassButton>
      </div>

      {!supported && (
        <p className="mt-2 text-xs text-amber-300">
          This browser has no speech synthesis — the sentence is shown but not
          spoken.
        </p>
      )}
    </section>
  );
}
