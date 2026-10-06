/**
 * Visible capture-state indicator: idle / recording / analysing.
 * @param {{state: "idle"|"recording"|"analysing"}} props
 */
export default function StatusBadge({ state }) {
  const styles = {
    idle: "bg-gray-700 text-gray-100",
    recording: "bg-red-600 text-white",
    analysing: "bg-amber-500 text-black",
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
      className={`inline-block rounded-full px-3 py-1 text-sm font-semibold ${styles[state]}`}
    >
      {label[state]}
    </span>
  );
}
