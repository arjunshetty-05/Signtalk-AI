import { useEffect, useMemo, useState } from "react";

// A tiny, dependency-free confetti burst. Bump the numeric `trigger` prop to
// fire a fresh burst (e.g. on each recognised/confirmed word). Pieces animate
// out and the overlay clears itself. Decorative + aria-hidden, and it renders
// nothing for users who prefer reduced motion.

const COLORS = ["#6366f1", "#a855f7", "#22d3ee", "#34d399", "#fbbf24", "#f472b6"];
const PIECES = 18;

function prefersReducedMotion() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export default function Celebrate({ trigger }) {
  const [burst, setBurst] = useState(0);
  const reduced = useMemo(prefersReducedMotion, []);

  useEffect(() => {
    if (!trigger) return; // skip the initial 0
    setBurst((b) => b + 1);
    const t = setTimeout(() => setBurst(0), 1100);
    return () => clearTimeout(t);
  }, [trigger]);

  // Precompute randomised trajectories per piece, re-rolled on each burst.
  const pieces = useMemo(() => {
    return Array.from({ length: PIECES }, (_, i) => {
      const angle = (Math.PI * (i / (PIECES - 1))) - Math.PI / 2; // -90°..+90°
      const dist = 90 + Math.random() * 120;
      return {
        id: i,
        dx: Math.sin(angle) * dist + (Math.random() - 0.5) * 40,
        dy: -Math.abs(Math.cos(angle)) * dist - Math.random() * 60,
        rot: (Math.random() - 0.5) * 540,
        color: COLORS[i % COLORS.length],
        delay: Math.random() * 60,
        size: 6 + Math.random() * 6,
        round: Math.random() > 0.5,
      };
    });
    // re-roll whenever a new burst starts
  }, [burst]);

  if (reduced || !burst) return null;

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 z-20 overflow-visible"
    >
      <div className="absolute left-1/2 top-1/2">
        {pieces.map((p) => (
          <span
            key={`${burst}-${p.id}`}
            className="confetti-piece absolute block"
            style={{
              width: `${p.size}px`,
              height: `${p.size}px`,
              background: p.color,
              borderRadius: p.round ? "9999px" : "2px",
              animationDelay: `${p.delay}ms`,
              // custom props consumed by the keyframes in index.css
              "--dx": `${p.dx}px`,
              "--dy": `${p.dy}px`,
              "--rot": `${p.rot}deg`,
            }}
          />
        ))}
      </div>
    </div>
  );
}
