/**
 * Visible capture-state indicator: idle / recording / analysing.
 * @param {{state: "idle"|"recording"|"analysing"}} props
 */
export default function StatusBadge({ state }) {
  const styles = {
    idle: "border-white/10 bg-white/5 text-slate-300",
    recording: "border-red-400/40 bg-red-500/15 text-red-200",
    analysing: "border-amber-400/40 bg-amber-400/15 text-amber-200",
  };
  const dot = {
    idle: "bg-slate-400",
    recording: "bg-red-400 animate-pulse",
    analysing: "bg-amber-300 animate-ping",
  };
  const label = {
    idle: "Ready",
    recording: "Recording",
    analysing: "Thinking…",
  };
  return (
    <span
      role="status"
      aria-live="polite"
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-semibold backdrop-blur-md ${styles[state]}`}
    >
      <span className={`h-2 w-2 rounded-full ${dot[state]}`} />
      {label[state]}
    </span>
  );
}
