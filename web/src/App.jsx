import { useMemo, useState } from "react";
import CaptureScreen from "./screens/CaptureScreen.jsx";
import EnrollScreen from "./screens/EnrollScreen.jsx";
import CaptionsScreen from "./screens/CaptionsScreen.jsx";

// Guest mode, no login wall (locked owner decision D3: no Firebase/auth). A
// stable per-browser guest id is enough to tag decisions and enrollment clips.
function useGuestId() {
  return useMemo(() => {
    const key = "signtalk_guest_id";
    let id = localStorage.getItem(key);
    if (!id) {
      id = `guest-${Math.random().toString(36).slice(2, 10)}`;
      localStorage.setItem(key, id);
    }
    return id;
  }, []);
}

const TABS = [
  { id: "capture", label: "Capture", icon: HandIcon },
  { id: "captions", label: "Captions", icon: WaveIcon },
  { id: "enroll", label: "Enrollment", icon: SparkIcon },
];

export default function App() {
  const [screen, setScreen] = useState("capture"); // capture | captions | enroll
  const signerId = useGuestId();

  const activeTab = TABS.find((t) => t.id === screen);

  return (
    <div className="relative min-h-full">
      {/* Ambient animated backdrop (decorative, behind everything). */}
      <div className="app-aurora" aria-hidden="true" />
      <div className="app-blobs" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>

      {/* SVG displacement filter that gives the liquid-glass buttons a real
          refracted/distorted backdrop (referenced by CSS: url(#lg-distortion)).
          Mounted once, invisible. */}
      <LiquidGlassFilter />

      {/* Sticky top bar: brand on the left, tab bar on the right. A max-width
          that matches the content keeps the two from drifting far apart. */}
      <header className="sticky top-0 z-20 border-b border-white/8 bg-[#0a0c14]/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-2xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-600 ring-1 ring-white/15">
              <HandIcon className="h-4 w-4 text-white" />
            </span>
            <span className="text-base font-bold tracking-tight text-white">
              SignTalk <span className="text-brand-400">AI</span>
            </span>
          </div>

          <nav
            aria-label="Screens"
            role="tablist"
            className="flex gap-0.5 rounded-xl border border-white/8 bg-white/[0.03] p-1"
          >
            {TABS.map((t) => (
              <TabButton
                key={t.id}
                active={screen === t.id}
                onClick={() => setScreen(t.id)}
                Icon={t.icon}
              >
                {t.label}
              </TabButton>
            ))}
          </nav>
        </div>
      </header>

      {/* Section 2.7 consent notice — always visible, one line. */}
      <p className="flex items-center justify-center gap-2 border-b border-amber-400/15 bg-amber-400/[0.07] px-4 py-2 text-center text-[13px] text-amber-200/90">
        <InfoIcon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        Your webcam video is sent to our server and may be stored for training
        and evaluation.
      </p>

      <main className="mx-auto max-w-2xl px-4 pb-16 pt-8">
        {/* Compact header for the active view — a title + one line of context,
            not a flashy marketing hero. */}
        <div className="mb-6 animate-fade-up">
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-brand-400">
            {activeTab && <activeTab.icon className="h-3.5 w-3.5" />}
            {activeTab?.label}
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-white">
            {SCREEN_META[screen].title}
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            {SCREEN_META[screen].subtitle}
          </p>
        </div>

        {/* `key` restarts the entrance animation on every tab switch. */}
        <div key={screen} className="animate-fade-up">
          {screen === "capture" && <CaptureScreen signerId={signerId} />}
          {screen === "captions" && <CaptionsScreen />}
          {screen === "enroll" && <EnrollScreen signerId={signerId} />}
        </div>
      </main>
    </div>
  );
}

// Per-screen heading + subtitle so each view opens with clear context.
const SCREEN_META = {
  capture: {
    title: "Sign to speech",
    subtitle: "Hold to record a sign, release, and we translate it to a spoken sentence.",
  },
  captions: {
    title: "Voice to captions",
    subtitle: "Speak and watch live captions build a conversation log.",
  },
  enroll: {
    title: "Enrollment",
    subtitle: "Record your own clips per sign to improve recognition accuracy.",
  },
};

/** A keyboard-operable, high-contrast pill tab with icon + sliding highlight. */
function TabButton({ active, onClick, children, Icon }) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`group relative flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] font-medium transition-colors duration-200 focus-visible:ring ${
        active ? "text-white" : "text-slate-400 hover:text-slate-200"
      }`}
    >
      {active && (
        <span
          aria-hidden="true"
          className="absolute inset-0 -z-10 rounded-lg bg-white/10 ring-1 ring-white/10"
        />
      )}
      <Icon className="h-4 w-4" />
      {children}
    </button>
  );
}

/**
 * Hidden SVG filter referenced by the `.lg-btn` backdrop-filter. feTurbulence
 * makes a soft noise field; feDisplacementMap uses it to bend whatever is
 * behind the glass by a few pixels, so edges refract like a real lens.
 */
function LiquidGlassFilter() {
  return (
    <svg
      aria-hidden="true"
      width="0"
      height="0"
      style={{ position: "absolute", pointerEvents: "none" }}
    >
      <defs>
        <filter id="lg-distortion" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.012 0.016"
            numOctaves="2"
            seed="7"
            result="noise"
          />
          <feGaussianBlur in="noise" stdDeviation="1.2" result="softNoise" />
          <feDisplacementMap
            in="SourceGraphic"
            in2="softNoise"
            scale="34"
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </defs>
    </svg>
  );
}

/* ---- Inline icons (no icon dependency) ---------------------------------- */
function HandIcon({ className = "h-4 w-4" }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M7 11V6.5a1.5 1.5 0 0 1 3 0V10m0 0V5a1.5 1.5 0 0 1 3 0v5m0-.5a1.5 1.5 0 0 1 3 0V11m0 0a1.5 1.5 0 0 1 3 0v3a7 7 0 0 1-7 7h-1.2a6 6 0 0 1-4.3-1.8l-3.1-3.2a1.6 1.6 0 0 1 2.3-2.2L7 15"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
function WaveIcon({ className = "h-4 w-4" }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M3 12h2m3-5v10m4-14v18m4-13v8m4-5h2"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}
function SparkIcon({ className = "h-4 w-4" }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 3l1.8 4.7L18.5 9l-4.7 1.8L12 15l-1.8-4.2L5.5 9l4.7-1.3L12 3zM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}
function InfoIcon({ className = "h-4 w-4" }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 11v5m0-8h.01" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}
