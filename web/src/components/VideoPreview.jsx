/**
 * Live webcam preview. The preview is CSS-mirrored (transform: scaleX(-1)) so
 * it feels like a mirror to the signer; the frames captured for upload are NOT
 * mirrored (that happens in useWebcam, drawn from the raw stream).
 *
 * @param {{videoRef: React.RefObject<HTMLVideoElement>, mirrored?: boolean,
 *          recording?: boolean, countdown?: number | null}} props
 */
export default function VideoPreview({ videoRef, mirrored = true, recording = false, countdown = null }) {
  return (
    <div
      className={`relative min-w-0 w-full overflow-hidden rounded-xl border bg-[#17211c] ${
        recording ? "border-[#c94e3f] ring-2 ring-[#c94e3f]/20" : "border-[#d8ddd7]"
      }`}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="block h-auto w-full max-w-full object-contain"
        style={{ transform: mirrored ? "scaleX(-1)" : "none", maxHeight: "min(55vh, 520px)" }}
      />
      {recording && (
        <span className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-[#a43d30] px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-white shadow">
          <span className="h-2 w-2 animate-pulse rounded-full bg-white" aria-hidden="true" />
          Recording
        </span>
      )}
      {countdown !== null && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/20 text-white" role="status" aria-live="assertive">
          <span className="flex h-24 w-24 items-center justify-center rounded-full border-4 border-white/90 bg-[#1c5b46]/90 text-5xl font-extrabold shadow-xl sm:h-28 sm:w-28 sm:text-6xl">
            {countdown}
          </span>
          <span className="mt-3 rounded-full bg-black/55 px-4 py-1.5 text-sm font-semibold">Get ready to sign</span>
        </div>
      )}
    </div>
  );
}
