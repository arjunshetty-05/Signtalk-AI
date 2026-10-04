// LanguageSelector.jsx — SignTalk AI web dashboard
//
// Dropdown for English/Hindi/Kannada. On change, calls POST /translate with
// the currently displayed sentence and reports the translated text back up
// so SubtitleBar re-renders in the selected language.

import { useState } from "react";
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

  const handleChange = async (event) => {
    const targetLang = event.target.value;
    setLanguage(targetLang);
    setError(null);

    if (!currentText) return;
    if (targetLang === "en") {
      onTranslated?.(currentText, targetLang);
      return;
    }

    setLoading(true);
    try {
      const { data } = await apiClient.post("/translate", {
        text: currentText,
        target_lang: targetLang,
        offline: false,
      });
      onTranslated?.(data.translated_text, targetLang);
    } catch (err) {
      setError("Translation failed — showing original text.");
      onTranslated?.(currentText, targetLang);
    } finally {
      setLoading(false);
    }
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
