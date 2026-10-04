/// tflite_inference_service.dart — SignTalk AI mobile (Offline mode)
///
/// ASSUMPTION — labels.json shape: this service assumes
/// assets/models/labels.json is a flat {"0": "HELLO", "1": "FOOD", ...}
/// mapping from the BiLSTM's output class index (as a string key) to the
/// human-readable sign label, matching train_bilstm.py's labels.json
/// export. If the real file ships with a different shape (e.g. nested
/// under a "classes" key), update ONLY `_loadLabels()` below — nothing
/// else in this file depends on the raw JSON structure.
///
/// Pipeline: CameraImage -> RGB -> movenet.tflite (17 keypoints) ->
/// normalize (mirrors keypoint_utils.py's shoulder-midpoint/bbox-diagonal
/// logic) -> 30-frame sliding buffer -> bilstm.tflite -> label.
///
/// Also routes translation through the bundled offline_phrasebook.json
/// asset instead of hitting the network, matching translation.py's
/// offline-mode phrasebook-only behavior.

import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'camera_utils.dart';

const int kMoveNetInputSize = 256;
const int kSequenceLength = 30;
const int kNumKeypoints = 17;
const int kLeftShoulder = 5, kRightShoulder = 6;
const int kLeftHip = 11, kRightHip = 12;
const double kConfidenceThreshold = 0.3;

class TFLiteInferenceService {
  Interpreter? _movenetInterpreter;
  Interpreter? _bilstmInterpreter;
  Map<String, String> _labels = {};
  Map<String, Map<String, String>> _phrasebook = {};

  final List<List<List<double>>> _sequenceBuffer = []; // (frame)(keypoint)(x,y)

  Future<void> initialize() async {
    _movenetInterpreter = await Interpreter.fromAsset('assets/models/movenet.tflite');
    _bilstmInterpreter = await Interpreter.fromAsset('assets/models/bilstm.tflite');
    await _loadLabels();
    await _loadPhrasebook();
  }

  Future<void> _loadLabels() async {
    final raw = await rootBundle.loadString('assets/models/labels.json');
    final decoded = jsonDecode(raw) as Map<String, dynamic>;
    _labels = decoded.map((k, v) => MapEntry(k, v.toString()));
  }

  Future<void> _loadPhrasebook() async {
    final raw = await rootBundle.loadString('assets/offline_phrasebook.json');
    final decoded = jsonDecode(raw) as Map<String, dynamic>;
    _phrasebook = decoded.map((phrase, translations) {
      final map = (translations as Map<String, dynamic>).map(
        (lang, text) => MapEntry(lang, text.toString()),
      );
      return MapEntry(phrase, map);
    });
  }

  /// Offline translation — phrasebook only, matches translate_text(offline=True).
  String translateOffline(String text, String targetLang) {
    final hit = _phrasebook[text.trim().toLowerCase()];
    if (hit != null && hit.containsKey(targetLang)) {
      return hit[targetLang]!;
    }
    return text; // no match — return original, same fallback as the backend
  }

  /// Feeds one camera frame through MoveNet -> normalize -> buffer ->
  /// (once full) BiLSTM. Returns a {"label", "confidence"} map once the
  /// 30-frame buffer fills and stabilizes, else null.
  Future<Map<String, dynamic>?> processFrame(CameraImage cameraImage) async {
    final rgb = convertCameraImageToRgb(cameraImage);
    final resized = img.copyResize(rgb, width: kMoveNetInputSize, height: kMoveNetInputSize);

    final input = _imageToInputTensor(resized);
    final output = List.generate(1, (_) => List.generate(1, (_) => List.generate(kNumKeypoints, (_) => List.filled(3, 0.0))));
    _movenetInterpreter!.run(input, output);

    final rawKeypoints = output[0][0]; // (17, 3) as [y, x, score]
    final normalized = _normalizeKeypoints(rawKeypoints);

    _sequenceBuffer.add(normalized);
    if (_sequenceBuffer.length > kSequenceLength) {
      _sequenceBuffer.removeAt(0);
    }
    if (_sequenceBuffer.length < kSequenceLength) return null;

    final flatInput = [
      _sequenceBuffer.map((frame) => frame.expand((xy) => xy).toList()).toList(),
    ]; // shape (1, 30, 34)

    final numClasses = _labels.length;
    final bilstmOutput = List.generate(1, (_) => List.filled(numClasses, 0.0));
    _bilstmInterpreter!.run(flatInput, bilstmOutput);

    final probs = bilstmOutput[0];
    var bestIdx = 0;
    var bestProb = probs.isNotEmpty ? probs[0] : 0.0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > bestProb) {
        bestProb = probs[i];
        bestIdx = i;
      }
    }

    return {
      'label': _labels[bestIdx.toString()] ?? 'class_$bestIdx',
      'confidence': bestProb,
    };
  }

  List<List<double>> _normalizeKeypoints(List<List<double>> rawKeypoints) {
    // rawKeypoints[i] = [y, x, score]
    final xy = rawKeypoints.map((kp) => [kp[1], kp[0]]).toList(); // -> (x, y)
    final scores = rawKeypoints.map((kp) => kp[2]).toList();
    final confident = scores.map((s) => s >= kConfidenceThreshold).toList();

    List<double> origin;
    if (confident[kLeftShoulder] && confident[kRightShoulder]) {
      origin = [
        (xy[kLeftShoulder][0] + xy[kRightShoulder][0]) / 2,
        (xy[kLeftShoulder][1] + xy[kRightShoulder][1]) / 2,
      ];
    } else {
      final confidentPts = [for (var i = 0; i < xy.length; i++) if (confident[i]) xy[i]];
      if (confidentPts.isNotEmpty) {
        origin = [
          confidentPts.map((p) => p[0]).reduce((a, b) => a + b) / confidentPts.length,
          confidentPts.map((p) => p[1]).reduce((a, b) => a + b) / confidentPts.length,
        ];
      } else {
        origin = [0.5, 0.5];
      }
    }

    final numConfident = confident.where((c) => c).length;
    double scale;
    if (numConfident >= 4) {
      final confidentPts = [for (var i = 0; i < xy.length; i++) if (confident[i]) xy[i]];
      final minX = confidentPts.map((p) => p[0]).reduce(min);
      final maxX = confidentPts.map((p) => p[0]).reduce(max);
      final minY = confidentPts.map((p) => p[1]).reduce(min);
      final maxY = confidentPts.map((p) => p[1]).reduce(max);
      scale = sqrt(pow(maxX - minX, 2) + pow(maxY - minY, 2)).toDouble();
    } else if (confident[kLeftShoulder] && confident[kRightShoulder] && confident[kLeftHip] && confident[kRightHip]) {
      final hipMid = [
        (xy[kLeftHip][0] + xy[kRightHip][0]) / 2,
        (xy[kLeftHip][1] + xy[kRightHip][1]) / 2,
      ];
      scale = sqrt(pow(hipMid[0] - origin[0], 2) + pow(hipMid[1] - origin[1], 2)).toDouble();
    } else {
      scale = 0.0;
    }
    if (scale < 1e-6) scale = 1.0;

    return xy.map((p) => [(p[0] - origin[0]) / scale, (p[1] - origin[1]) / scale]).toList();
  }

  List<List<List<List<double>>>> _imageToInputTensor(img.Image image) {
    return [
      List.generate(
        kMoveNetInputSize,
        (y) => List.generate(kMoveNetInputSize, (x) {
          final pixel = image.getPixel(x, y);
          return [pixel.r.toDouble(), pixel.g.toDouble(), pixel.b.toDouble()];
        }),
      ),
    ];
  }

  void resetSequence() => _sequenceBuffer.clear();

  void dispose() {
    _movenetInterpreter?.close();
    _bilstmInterpreter?.close();
  }
}
