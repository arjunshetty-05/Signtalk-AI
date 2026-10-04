/// websocket_service.dart — SignTalk AI mobile
///
/// Wraps /ws/gesture and /ws/speech behind web_socket_channel, mirroring
/// the web dashboard's useGestureSocket hook: connect/reconnect/disconnect
/// lifecycle isolated in one place, auth passed as ?token=<jwt> (mobile
/// WebSocket clients hit the same limitation as browsers here — keeping the
/// same auth mechanism keeps both clients consistent with one backend
/// contract).

import 'dart:async';
import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/gesture_event.dart';
import 'api_service.dart';

const _reconnectDelay = Duration(seconds: 2);

class GestureWebSocketService {
  final String token;
  final int frameSkip;
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  bool _shouldReconnect = true;

  final _labelController = StreamController<GestureEvent>.broadcast();
  final _sentenceController = StreamController<CorrectedSentenceEvent>.broadcast();
  final _connectionController = StreamController<bool>.broadcast();

  Stream<GestureEvent> get labelStream => _labelController.stream;
  Stream<CorrectedSentenceEvent> get sentenceStream => _sentenceController.stream;
  Stream<bool> get connectionStream => _connectionController.stream;

  GestureWebSocketService({required this.token, this.frameSkip = 1});

  String get _wsBaseUrl {
    final httpBase = ApiService.baseUrl;
    return httpBase.replaceFirst('http', 'ws');
  }

  void connect() {
    _shouldReconnect = true;
    final uri = Uri.parse('$_wsBaseUrl/ws/gesture?token=$token&frame_skip=$frameSkip');
    _channel = WebSocketChannel.connect(uri);
    _connectionController.add(true);

    _subscription = _channel!.stream.listen(
      (message) => _handleMessage(message as String),
      onDone: _handleDisconnect,
      onError: (_) => _handleDisconnect(),
    );
  }

  void _handleMessage(String raw) {
    try {
      final data = jsonDecode(raw) as Map<String, dynamic>;
      if (data.containsKey('error')) return;
      if (data['type'] == 'corrected_sentence') {
        _sentenceController.add(CorrectedSentenceEvent.fromJson(data));
      } else if (data.containsKey('label')) {
        _labelController.add(GestureEvent.fromJson(data));
      }
    } catch (_) {
      // malformed frame — ignore
    }
  }

  void _handleDisconnect() {
    _connectionController.add(false);
    if (_shouldReconnect) {
      Future.delayed(_reconnectDelay, connect);
    }
  }

  void sendFrame(String base64Jpeg) {
    _channel?.sink.add(jsonEncode({'frame': base64Jpeg}));
  }

  void dispose() {
    _shouldReconnect = false;
    _subscription?.cancel();
    _channel?.sink.close();
    _labelController.close();
    _sentenceController.close();
    _connectionController.close();
  }
}

class SpeechWebSocketService {
  final String token;
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;

  final _transcriptController = StreamController<SpeechTranscript>.broadcast();
  Stream<SpeechTranscript> get transcriptStream => _transcriptController.stream;

  SpeechWebSocketService({required this.token});

  void connect() {
    final httpBase = ApiService.baseUrl;
    final wsBase = httpBase.replaceFirst('http', 'ws');
    final uri = Uri.parse('$wsBase/ws/speech?token=$token');
    _channel = WebSocketChannel.connect(uri);

    _subscription = _channel!.stream.listen((message) {
      try {
        final data = jsonDecode(message as String) as Map<String, dynamic>;
        _transcriptController.add(SpeechTranscript.fromJson(data));
      } catch (_) {
        // ignore malformed frame
      }
    });
  }

  void sendAudioChunk(List<int> pcmBytes) {
    _channel?.sink.add(pcmBytes);
  }

  void dispose() {
    _subscription?.cancel();
    _channel?.sink.close();
    _transcriptController.close();
  }
}
