import { useMemo, useState } from "react";
import CaptureScreen from "./screens/CaptureScreen.jsx";
import EnrollScreen from "./screens/EnrollScreen.jsx";
import CaptionsScreen from "./screens/CaptionsScreen.jsx";
import SignGuideScreen from "./screens/SignGuideScreen.jsx";

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

export default function App() {
  const [screen, setScreen] = useState("capture"); // capture | captions | enroll | guide
  const signerId = useGuestId();

  return (
    <div className="min-h-full">
      <a
        href="#main"
        className="sr-only z-50 rounded-lg bg-white px-4 py-3 text-[#1c5b46] focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus-visible:ring"
      >
        Skip to content
      </a>
      <header className="border-b border-[#e6e3db] bg-[#fbfaf7]">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-2.5 sm:flex-row sm:items-center sm:justify-between sm:px-6 sm:py-3 lg:px-8">
          <a href="#main" className="flex items-center gap-3 rounded-lg focus-visible:ring">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-[#1c5b46] text-sm font-bold tracking-tight text-white">
              ST
            </span>
            <span>
              <span className="block text-lg font-bold tracking-tight text-[#1d2c25]">
                SignTalk
              </span>
              <span className="hidden text-[11px] font-semibold uppercase tracking-[0.16em] text-[#78847c] md:block">
                Sign · speech · understanding
              </span>
            </span>
          </a>
          <nav
            aria-label="Screens"
            className="flex w-full gap-0.5 overflow-x-auto rounded-xl bg-[#f0eee8] p-1 sm:w-auto sm:gap-1"
            role="tablist"
          >
            <TabButton active={screen === "capture"} onClick={() => setScreen("capture")}>
              Capture
            </TabButton>
            <TabButton active={screen === "captions"} onClick={() => setScreen("captions")}>
              Voice captions
            </TabButton>
            <TabButton active={screen === "guide"} onClick={() => setScreen("guide")}>
              Sign guide
            </TabButton>
            <TabButton active={screen === "enroll"} onClick={() => setScreen("enroll")}>
              Enrollment
            </TabButton>
          </nav>
        </div>
      </header>

      <div className="border-b border-[#ead9b9] bg-[#fff8e9]">
        <p className="mx-auto flex max-w-7xl items-center gap-2 px-4 py-1.5 text-xs text-[#76551e] sm:px-6 sm:py-2 sm:text-sm lg:px-8">
          <span aria-hidden="true" className="text-base">●</span>
          Your webcam video is sent to our server and may be stored for training and evaluation.
        </p>
      </div>

      <main id="main" className="mx-auto max-w-7xl px-4 py-4 sm:px-6 sm:py-6 lg:px-8">
        {screen === "capture" && <CaptureScreen signerId={signerId} />}
        {screen === "captions" && <CaptionsScreen />}
        {screen === "guide" && <SignGuideScreen onPractice={() => setScreen("capture")} />}
        {screen === "enroll" && <EnrollScreen signerId={signerId} />}
      </main>
    </div>
  );
}

/** A keyboard-operable, high-contrast tab button. */
function TabButton({ active, onClick, children }) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`shrink-0 rounded-lg px-2.5 py-2 text-xs font-semibold transition-colors focus-visible:ring sm:px-3 sm:text-sm ${
        active
          ? "bg-white text-[#1c5b46] shadow-sm"
          : "text-[#66736b] hover:bg-[#e8e5dd] hover:text-[#26362e]"
      }`}
    >
      {children}
    </button>
  );
}
