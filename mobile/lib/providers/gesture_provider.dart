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

  // Cold start races the default `false` (fireImmediately) against the
  // async SharedPreferences restore of the real persisted value — two
  // overlapping _switchMode calls can otherwise interleave and leave the
  // wrong service connected. Each call captures its own generation and
  // bails if a newer call has since started, instead of letting whichever
  // one *finishes* last win regardless of which was requested last.
  int _switchGeneration = 0;

  GestureNotifier(this.ref) : super(const GestureState()) {
    ref.listen<bool>(offlineModeProvider, (previous, next) => _switchMode(next), fireImmediately: true);
  }

  Future<void> _switchMode(bool offline) async {
    final generation = ++_switchGeneration;
    _offline = offline;
    _onlineService?.dispose();
    _onlineService = null;
    _offlineService?.dispose();
    _offlineService = null;

    if (offline) {
      final service = TFLiteInferenceService();
      await service.initialize();
      if (generation != _switchGeneration) {
        service.dispose();
        return;
      }
      _offlineService = service;
      state = state.copyWith(connected: true);
    } else {
      final authController = ref.read(authControllerProvider);
      final service = GestureWebSocketService(getToken: authController.currentToken);
      service.connect();
      if (generation != _switchGeneration) {
        service.dispose();
        return;
      }
      _onlineService = service;
      service.labelStream.listen((event) {
        if (generation != _switchGeneration) return;
        state = state.copyWith(latestLabel: event);
      });
      service.sentenceStream.listen((event) {
        if (generation != _switchGeneration) return;
        state = state.copyWith(latestSentence: event);
      });
      service.connectionStream.listen((connected) {
        if (generation != _switchGeneration) return;
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
