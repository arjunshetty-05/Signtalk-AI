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
  const [compositionError, setCompositionError] = useState("");
  const [demoWords, setDemoWords] = useState([]);
  const idleTimer = useRef(null);
  const requestId = useRef(0);
  const { speak, speaking, cancel, supported, voiceFor } = useSpeech();
  const sentenceWords = words.length ? words : demoWords;

  const doCompose = useCallback(async (wordsToCompose = words) => {
    if (!wordsToCompose.length) return;
    const currentRequest = ++requestId.current;
    setComposing(true);
    setCompositionError("");
    try {
      const data = await composeSentence({ words: wordsToCompose });
      if (currentRequest === requestId.current) setComposed(data);
    } catch (error) {
      console.error("Sentence composition failed:", error);
      if (currentRequest === requestId.current) {
        setComposed(null);
        setCompositionError("Could not translate this sentence. Please try again.");
      }
    } finally {
      if (currentRequest === requestId.current) setComposing(false);
    }
  }, [words]);

  // Flush on idle gap or max words: compose (but don't auto-speak in strict mode).
  useEffect(() => {
    if (idleTimer.current) clearTimeout(idleTimer.current);
    if (words.length && demoWords.length) setDemoWords([]);
    if (!words.length) {
      if (demoWords.length) {
        doCompose(demoWords);
        return undefined;
      }
      requestId.current += 1;
      setComposed(null);
      setCompositionError("");
      setComposing(false);
      return undefined;
    }
    setComposed(null);
    setCompositionError("");
    if (sentenceWords.length >= MAX_WORDS) {
      doCompose();
      return undefined;
    }
    idleTimer.current = setTimeout(doCompose, IDLE_MS);
    return () => idleTimer.current && clearTimeout(idleTimer.current);
  }, [words, demoWords, sentenceWords, doCompose]);

  const sentence = composed?.sentences?.[lang] || "";
  const selectedVoice = voiceFor(lang);

  return (
    <section className="rounded-2xl border border-[#e6e3db] bg-white p-3 shadow-[0_8px_30px_rgba(36,49,40,0.05)] sm:p-4" aria-label="Sentence">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-[#8b6a38]">Your message</p>
          <h3 className="mt-0.5 text-lg font-bold text-[#26362e]">Sentence</h3>
        </div>
        <div className="flex gap-1 rounded-lg bg-[#f2f1ec] p-1" role="group" aria-label="Language">
          {APP_LANGS.map((l) => (
            <button
              key={l.code}
              type="button"
              onClick={() => setLang(l.code)}
              aria-pressed={lang === l.code}
              className={`rounded-md px-3 py-2 text-xs font-semibold transition-colors sm:text-sm ${
                lang === l.code ? "bg-white text-[#1c5b46] shadow-sm" : "text-[#68766e] hover:text-[#26362e]"
              }`}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>

      {/* Running word chips */}
      <div className="mb-2 flex min-h-[1.5rem] flex-wrap gap-2">
        {sentenceWords.length ? (
          sentenceWords.map((w, i) => (
            <span key={`${w}-${i}`} className="rounded-full border border-[#d9e6dc] bg-[#f0f6f0] px-3 py-1 text-sm font-medium text-[#356449]">
              {w}
            </span>
          ))
        ) : (
          <span className="text-sm text-[#8a948c]">
            Accepted words collect here, then become a sentence.
          </span>
        )}
      </div>

      <div className="mb-3 rounded-xl bg-[#f8f7f3] p-2.5">
        <p className="mb-2 text-xs font-semibold text-[#68766e]">Try a demo sentence</p>
        <div className="flex flex-wrap gap-2">
          {[
            { label: "Hello", words: ["hello"] },
            { label: "Thank you", words: ["thank you"] },
            { label: "Help please", words: ["help", "please"] },
            {
              label: "Long: ask for water",
              words: ["hello", "water", "please", "help", "thank you"],
            },
            {
              label: "Long: explain a problem",
              words: ["pain", "help", "water", "please"],
            },
          ].map((demo) => (
            <button
              key={demo.label}
              type="button"
              disabled={words.length > 0 || composing}
              onClick={() => setDemoWords(demo.words)}
              className="rounded-lg border border-[#dddcd4] bg-white px-2.5 py-1.5 text-xs font-medium text-[#4c5b52] transition hover:border-[#a9c4ad] hover:bg-[#f0f6f0] focus-visible:ring disabled:opacity-50 sm:text-sm"
            >
              {demo.label}
            </button>
          ))}
        </div>
      </div>

      {/* Composed sentence (large, high-contrast subtitle) */}
      <p
        className="min-h-[2.5rem] text-lg font-semibold leading-relaxed text-[#23352b] sm:text-xl"
        aria-live="polite"
        lang={lang}
      >
        {composing ? "…" : sentence}
      </p>
      {compositionError && (
        <p className="mt-2 rounded-lg bg-[#fff8e9] px-3 py-2 text-sm text-[#916b2c]" role="status">
          {compositionError}
        </p>
      )}
      {composed && (
        <p className="mt-2 text-xs text-[#7b877e]">
          source: {composed.source}
          {demoWords.length || !composed.verified
            ? " (unverified — review before trusting)"
            : " (verified)"}
        </p>
      )}

      <div className="mt-2 flex flex-wrap gap-2">
        <button
          type="button"
          disabled={!sentence || !supported || !selectedVoice || speaking}
          onClick={() => speak(sentence, lang)}
          className="rounded-lg bg-[#1c5b46] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#164b39] focus-visible:ring disabled:opacity-50"
        >
          Speak
        </button>
        {speaking && (
          <button type="button" onClick={cancel} className="rounded-lg border border-[#dddcd4] px-4 py-2 text-sm font-semibold text-[#52635a] hover:bg-[#f5f4ef]">
            Stop
          </button>
        )}
        <button
          type="button"
          disabled={!sentenceWords.length}
          onClick={() => {
            cancel();
            setDemoWords([]);
            setComposed(null);
            setCompositionError("");
            onClear();
          }}
          className="rounded-lg border border-[#dddcd4] px-4 py-2 text-sm font-semibold text-[#52635a] transition hover:bg-[#f5f4ef] focus-visible:ring disabled:opacity-50"
        >
          Finish / Clear
        </button>
      </div>

      {!supported && (
        <p className="mt-3 rounded-lg bg-[#fff8e9] px-3 py-2 text-xs leading-relaxed text-[#916b2c]">
          This browser has no speech synthesis — the sentence is shown but not spoken.
        </p>
      )}
      {supported && sentence && !selectedVoice && (
        <p className="mt-3 rounded-lg bg-[#fff8e9] px-3 py-2 text-xs leading-relaxed text-[#916b2c]" role="status">
          No {APP_LANGS.find((item) => item.code === lang)?.label} speech voice is
          available in this browser. Add that language&apos;s speech voice in your
          device settings, then reload this page.
        </p>
      )}
    </section>
  );
}
