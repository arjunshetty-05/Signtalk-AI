/**
 * Live webcam preview. The preview is CSS-mirrored (transform: scaleX(-1)) so
 * it feels like a mirror to the signer; the frames captured for upload are NOT
 * mirrored (that happens in useWebcam, drawn from the raw stream).
 *
 * @param {{videoRef: React.RefObject<HTMLVideoElement>, mirrored?: boolean,
 *          recording?: boolean}} props
 */
export default function VideoPreview({ videoRef, mirrored = true, recording = false }) {
  return (
    <div
      className={`relative rounded-2xl p-[2px] transition-all duration-300 ${
        recording
          ? "bg-gradient-to-br from-red-500 via-rose-500 to-orange-400 shadow-[0_0_40px_-6px_rgba(239,68,68,0.6)]"
          : "bg-gradient-to-br from-brand-500/60 via-accent-violet/50 to-accent-cyan/60 shadow-glow"
      }`}
    >
      <div className="relative overflow-hidden rounded-[14px] bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-auto w-full"
          style={{ transform: mirrored ? "scaleX(-1)" : "none" }}
        />

        {/* Corner framing guides for nicer composition. */}
        <Corners />

        {recording && (
          <span className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-red-600/90 px-3 py-1 text-sm font-semibold text-white shadow-lg backdrop-blur-sm animate-pulse-ring">
            <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
            REC
          </span>
        )}

        {/* Soft vignette so the subject reads clearly. */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 rounded-[14px] shadow-[inset_0_0_80px_rgba(0,0,0,0.45)]"
        />
      </div>
    </div>
  );
}

/** Decorative viewfinder corner brackets. */
function Corners() {
  const base =
    "pointer-events-none absolute h-6 w-6 border-white/40 transition-opacity";
  return (
    <div aria-hidden="true" className="opacity-70">
      <span className={`${base} left-3 top-3 rounded-tl-md border-l-2 border-t-2`} />
      <span className={`${base} right-3 top-3 rounded-tr-md border-r-2 border-t-2`} />
      <span className={`${base} bottom-3 left-3 rounded-bl-md border-b-2 border-l-2`} />
      <span className={`${base} bottom-3 right-3 rounded-br-md border-b-2 border-r-2`} />
    </div>
  );
}
