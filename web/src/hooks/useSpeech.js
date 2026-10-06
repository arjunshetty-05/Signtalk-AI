import { useCallback, useEffect, useState } from "react";

// Browser SpeechSynthesis TTS (PROJECT_CONTEXT 5.12): free, offline-capable,
// the default speech path. Hindi/Kannada voice availability varies by device,
// so this must be tested on the demo machine; we pick the best matching voice
// per language and fall back gracefully if none exists.
const LANG_CODES = { en: "en-IN", hi: "hi-IN", kn: "kn-IN" };
const LANG_FALLBACK = { en: "en", hi: "hi", kn: "kn" };

/**
 * Speak text in a given app language ("en" | "hi" | "kn") using the browser's
 * built-in voices. Returns `{ speak, speaking, cancel, voiceFor, supported }`.
 */
export function useSpeech() {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const [voices, setVoices] = useState([]);
  const [speaking, setSpeaking] = useState(false);

  useEffect(() => {
    if (!supported) return undefined;
    const load = () => setVoices(window.speechSynthesis.getVoices());
    load();
    window.speechSynthesis.onvoiceschanged = load;
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, [supported]);

  /** Best available voice for an app language, or null if none matches. */
  const voiceFor = useCallback(
    (lang) => {
      if (!voices.length) return null;
      const code = LANG_CODES[lang] || lang;
      const fb = LANG_FALLBACK[lang] || lang;
      return (
        voices.find((v) => v.lang === code) ||
        voices.find((v) => v.lang?.toLowerCase().startsWith(fb)) ||
        null
      );
    },
    [voices]
  );

  const cancel = useCallback(() => {
    if (supported) window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  /** Speak `text` in app language `lang`. No-op (returns false) if unsupported. */
  const speak = useCallback(
    (text, lang = "en") => {
      if (!supported || !text) return false;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      const v = voiceFor(lang);
      if (v) u.voice = v;
      u.lang = LANG_CODES[lang] || lang;
      u.onend = () => setSpeaking(false);
      u.onerror = () => setSpeaking(false);
      setSpeaking(true);
      window.speechSynthesis.speak(u);
      return true;
    },
    [supported, voiceFor]
  );

  return { speak, speaking, cancel, voiceFor, supported };
}
