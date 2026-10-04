// SubtitleBar.jsx — SignTalk AI web dashboard
//
// Caption-style bar fixed to the bottom of the screen, showing the
// corrected full-sentence output. Dedupes on sentence text + receivedAt
// timestamp (a sparse, not continuous, event) so the fade/slide-in
// animation only fires for genuinely new sentences.

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function SubtitleBar({ sentenceEvent }) {
  const [displayed, setDisplayed] = useState(null);
  const lastKeyRef = useRef(null);

  useEffect(() => {
    if (!sentenceEvent?.sentence) return;
    const key = `${sentenceEvent.sentence}|${sentenceEvent.receivedAt}`;
    if (key === lastKeyRef.current) return; // dedupe — not a genuinely new event
    lastKeyRef.current = key;
    setDisplayed(sentenceEvent);
  }, [sentenceEvent]);

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 w-full max-w-2xl px-4 z-20">
      <AnimatePresence mode="wait">
        {displayed && (
          <motion.div
            key={`${displayed.sentence}|${displayed.receivedAt}`}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="rounded-xl bg-black/70 backdrop-blur-md border border-white/10 px-5 py-3 text-center"
          >
            <p className="text-white text-lg font-medium">{displayed.sentence}</p>
            <div className="flex items-center justify-center gap-2 mt-1 text-[11px] text-neutral-400">
              <span>{displayed.source === "gemini" ? "Gemini 2.0 Flash" : "Flan-T5 (offline)"}</span>
              {displayed.low_confidence && (
                <span className="text-amber-400 font-semibold">Low confidence</span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
