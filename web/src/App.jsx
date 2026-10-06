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

export default function App() {
  const [screen, setScreen] = useState("capture"); // capture | enroll
  const signerId = useGuestId();

  return (
    <div className="min-h-full">
      <header className="border-b border-gray-800 bg-black/40">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
          <h1 className="text-lg font-bold">SignTalk AI</h1>
          <nav aria-label="Screens" className="flex gap-1" role="tablist">
            <TabButton
              active={screen === "capture"}
              onClick={() => setScreen("capture")}
            >
              Capture
            </TabButton>
            <TabButton
              active={screen === "captions"}
              onClick={() => setScreen("captions")}
            >
              Captions
            </TabButton>
            <TabButton
              active={screen === "enroll"}
              onClick={() => setScreen("enroll")}
            >
              Enrollment
            </TabButton>
          </nav>
        </div>
      </header>

      {/* Section 2.7 consent notice — always visible, one line. */}
      <p className="bg-amber-950/60 px-4 py-2 text-center text-sm text-amber-100">
        Your webcam video is sent to our server and may be stored for training
        and evaluation.
      </p>

      <main className="px-4 py-6">
        {screen === "capture" && <CaptureScreen signerId={signerId} />}
        {screen === "captions" && <CaptionsScreen />}
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
      className={`rounded px-3 py-1.5 text-sm font-semibold focus-visible:ring ${
        active
          ? "bg-blue-600 text-white"
          : "bg-gray-800 text-gray-200 hover:bg-gray-700"
      }`}
    >
      {children}
    </button>
  );
}
