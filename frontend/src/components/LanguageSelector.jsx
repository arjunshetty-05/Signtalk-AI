// LanguageSelector.jsx — SignTalk AI web dashboard
//
// Dropdown for English/Hindi/Kannada. On change, calls POST /translate with
// the currently displayed sentence and reports the translated text back up
// so SubtitleBar re-renders in the selected language.

import { useState, useEffect, useCallback } from "react";
import { apiClient } from "../context/AuthProvider.jsx";

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "hi", label: "हिन्दी" },
  { code: "kn", label: "ಕನ್ನಡ" },
];

export default function LanguageSelector({ currentText, onTranslated }) {
  const [language, setLanguage] = useState("en");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const translate = useCallback(async (text, targetLang) => {
    setError(null);
    if (!text) return;

    if (targetLang === "en") {
      onTranslated?.(text, targetLang);
      return;
    }

    setLoading(true);
    try {
      const { data } = await apiClient.post("/translate", {
        text,
        target_lang: targetLang,
        offline: false,
      });
      onTranslated?.(data.translated_text, targetLang);
    } catch (err) {
      setError("Translation failed — showing original text.");
      onTranslated?.(text, targetLang);
    } finally {
      setLoading(false);
    }
  }, [onTranslated]);

  // Re-translate automatically whenever a new sentence arrives while a
  // non-English language is selected. Translation used to only fire from
  // the dropdown's onChange, so every sentence after the first one silently
  // reverted to English.
  useEffect(() => {
    translate(currentText, language);
    // Only re-run when the sentence itself changes — language changes are
    // handled by handleChange below, to avoid a duplicate call.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentText]);

  const handleChange = (event) => {
    const targetLang = event.target.value;
    setLanguage(targetLang);
    translate(currentText, targetLang);
  };

  return (
    <div className="flex items-center gap-2">
      <select
        value={language}
        onChange={handleChange}
        className="glass-panel rounded-lg px-3 py-1.5 text-sm text-white bg-transparent outline-none"
      >
        {LANGUAGES.map((lang) => (
          <option key={lang.code} value={lang.code} className="bg-neutral-900">
            {lang.label}
          </option>
        ))}
      </select>
      {loading && <span className="text-xs text-neutral-400">Translating...</span>}
      {error && <span className="text-xs text-red-400">{error}</span>}
    </div>
  );
}
