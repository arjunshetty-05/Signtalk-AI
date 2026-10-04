/// gesture_provider.dart — SignTalk AI mobile
///
/// Riverpod-managed gesture pipeline state. Switches between the live
/// GestureWebSocketService (online) and TFLiteInferenceService (offline)
/// based on offlineModeProvider — screens just read `gestureStateProvider`
/// and call `sendFrame`/`processOfflineFrame` without caring which path is
/// active.

import 'package:camera/camera.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/gesture_event.dart';
import '../services/tflite_inference_service.dart';
import '../services/websocket_service.dart';
import 'auth_provider.dart';
import 'offline_mode_provider.dart';

class GestureState {
  final GestureEvent? latestLabel;
  final CorrectedSentenceEvent? latestSentence;
  final bool connected;

  const GestureState({this.latestLabel, this.latestSentence, this.connected = false});

  GestureState copyWith({
    GestureEvent? latestLabel,
    CorrectedSentenceEvent? latestSentence,
    bool? connected,
  }) {
    return GestureState(
      latestLabel: latestLabel ?? this.latestLabel,
      latestSentence: latestSentence ?? this.latestSentence,
      connected: connected ?? this.connected,
    );
  }
}

class GestureNotifier extends StateNotifier<GestureState> {
  final Ref ref;
  GestureWebSocketService? _onlineService;
  TFLiteInferenceService? _offlineService;
  bool _offline = false;

  GestureNotifier(this.ref) : super(const GestureState()) {
    ref.listen<bool>(offlineModeProvider, (previous, next) => _switchMode(next), fireImmediately: true);
  }

  Future<void> _switchMode(bool offline) async {
    _offline = offline;
    _onlineService?.dispose();
    _onlineService = null;
    _offlineService?.dispose();
    _offlineService = null;

    if (offline) {
      _offlineService = TFLiteInferenceService();
      await _offlineService!.initialize();
      state = state.copyWith(connected: true);
    } else {
      final authController = ref.read(authControllerProvider);
      _onlineService = GestureWebSocketService(getToken: authController.currentToken)..connect();
      _onlineService!.labelStream.listen((event) {
        state = state.copyWith(latestLabel: event);
      });
      _onlineService!.sentenceStream.listen((event) {
        state = state.copyWith(latestSentence: event);
      });
      _onlineService!.connectionStream.listen((connected) {
        state = state.copyWith(connected: connected);
      });
    }
  }

  void sendFrame(String base64Jpeg) {
    if (!_offline) _onlineService?.sendFrame(base64Jpeg);
  }

  Future<void> processOfflineFrame(CameraImage cameraImage) async {
    if (!_offline || _offlineService == null) return;
    final result = await _offlineService!.processFrame(cameraImage);
    if (result != null) {
      state = state.copyWith(
        latestLabel: GestureEvent(
          label: result['label'] as String,
          confidence: (result['confidence'] as double),
          timestamp: DateTime.now().millisecondsSinceEpoch / 1000,
        ),
      );
    }
  }

  @override
  void dispose() {
    _onlineService?.dispose();
    _offlineService?.dispose();
    super.dispose();
  }
}

final gestureStateProvider = StateNotifierProvider<GestureNotifier, GestureState>((ref) {
  return GestureNotifier(ref);
});
