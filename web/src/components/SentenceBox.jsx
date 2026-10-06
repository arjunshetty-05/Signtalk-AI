import { useCallback, useEffect, useRef, useState } from "react";
import { composeSentence } from "../api.js";
import { useSpeech } from "../hooks/useSpeech.js";

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
    <section className="mt-8 rounded-lg border border-gray-700 p-4" aria-label="Sentence">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-bold">Sentence</h3>
        <div className="flex gap-1" role="group" aria-label="Language">
          {APP_LANGS.map((l) => (
            <button
              key={l.code}
              type="button"
              onClick={() => setLang(l.code)}
              aria-pressed={lang === l.code}
              className={`rounded px-3 py-1 text-sm ${
                lang === l.code ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-200"
              }`}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>

      {/* Running word chips */}
      <div className="mb-3 flex min-h-[2rem] flex-wrap gap-2">
        {words.length ? (
          words.map((w, i) => (
            <span key={`${w}-${i}`} className="rounded-full bg-gray-700 px-3 py-1 text-sm">
              {w}
            </span>
          ))
        ) : (
          <span className="text-sm text-gray-500">
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
        {composing ? "…" : sentence}
      </p>
      {composed && (
        <p className="mt-1 text-xs text-gray-400">
          source: {composed.source}
          {composed.verified ? " (verified)" : " (unverified — review before trusting)"}
        </p>
      )}

      <div className="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          disabled={!sentence || !supported || speaking}
          onClick={() => speak(sentence, lang)}
          className="rounded bg-green-600 px-4 py-2 font-semibold text-white hover:bg-green-500 focus-visible:ring disabled:opacity-50"
        >
          🔊 Speak
        </button>
        {speaking && (
          <button type="button" onClick={cancel} className="rounded bg-gray-700 px-4 py-2">
            Stop
          </button>
        )}
        <button
          type="button"
          disabled={!words.length}
          onClick={() => {
            cancel();
            setComposed(null);
            onClear();
          }}
          className="rounded bg-gray-700 px-4 py-2 hover:bg-gray-600 focus-visible:ring disabled:opacity-50"
        >
          Finish / Clear
        </button>
      </div>

      {!supported && (
        <p className="mt-2 text-xs text-amber-300">
          This browser has no speech synthesis — the sentence is shown but not spoken.
        </p>
      )}
    </section>
  );
}
