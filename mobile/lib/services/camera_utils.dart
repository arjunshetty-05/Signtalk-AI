/// camera_utils.dart — SignTalk AI mobile
///
/// Shared CameraImage (YUV420) -> img.Image (RGB) conversion, reused by
/// both the online path (home_camera_screen.dart, JPEG-encodes this for
/// the WebSocket) and the offline path (tflite_inference_service.dart,
/// feeds this straight into MoveNet) — do not duplicate this logic.

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

img.Image convertCameraImageToRgb(CameraImage cameraImage) {
  final width = cameraImage.width;
  final height = cameraImage.height;
  final image = img.Image(width: width, height: height);

  final yPlane = cameraImage.planes[0];
  final uPlane = cameraImage.planes[1];
  final vPlane = cameraImage.planes[2];

  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final uvIndex = (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2) * (uPlane.bytesPerPixel ?? 1);
      final yIndex = y * yPlane.bytesPerRow + x;

      final yValue = yPlane.bytes[yIndex];
      final uValue = uPlane.bytes[uvIndex];
      final vValue = vPlane.bytes[uvIndex];

      final r = (yValue + 1.370705 * (vValue - 128)).clamp(0, 255).toInt();
      final g = (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128)).clamp(0, 255).toInt();
      final b = (yValue + 1.732446 * (uValue - 128)).clamp(0, 255).toInt();

      image.setPixelRgb(x, y, r, g, b);
    }
  }
  return image;
}
