/**
 * Visible capture-state indicator: idle / recording / analysing.
 * @param {{state: "idle"|"recording"|"analysing"}} props
 */
export default function StatusBadge({ state }) {
  const styles = {
    idle: "border border-[#dfe4dc] bg-[#edf1eb] text-[#52665a]",
    recording: "bg-[#fae8e3] text-[#a43d30]",
    analysing: "bg-[#fff3d8] text-[#8b6425]",
  };
  const label = {
    idle: "Idle",
    recording: "Recording…",
    analysing: "Analysing…",
  };
  return (
    <span
      role="status"
      aria-live="polite"
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-bold ${styles[state]}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${state === "recording" ? "animate-pulse bg-[#c94e3f]" : state === "analysing" ? "bg-[#c08a31]" : "bg-[#639273]"}`} aria-hidden="true" />
      {label[state]}
    </span>
  );
}
