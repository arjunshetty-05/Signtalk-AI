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
      className={`relative overflow-hidden rounded-lg border-4 bg-black ${
        recording ? "border-red-500" : "border-gray-700"
      }`}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="h-auto w-full"
        style={{ transform: mirrored ? "scaleX(-1)" : "none" }}
      />
      {recording && (
        <span className="absolute left-3 top-3 flex items-center gap-2 rounded bg-red-600 px-2 py-1 text-sm font-semibold text-white">
          <span className="h-2 w-2 animate-pulse rounded-full bg-white" />
          REC
        </span>
      )}
    </div>
  );
}
