export async function openWideWebcam({ width = 640, height = 480, deviceId } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: deviceId
      ? { width, height, deviceId: { exact: deviceId } }
      : { width, height, facingMode: "user" },
    audio: false,
  });
  const track = stream.getVideoTracks()[0];

  try {
    const zoom = track.getCapabilities?.().zoom;
    if (!zoom || typeof zoom.min !== "number") {
      return { stream, wideView: false };
    }

    const currentZoom = track.getSettings?.().zoom;
    if (typeof currentZoom === "number" && currentZoom <= zoom.min) {
      return { stream, wideView: true };
    }

    await track.applyConstraints({ advanced: [{ zoom: zoom.min }] });
    return { stream, wideView: true };
  } catch {
    return { stream, wideView: false };
  }
}
