/// gesture_event.dart — SignTalk AI mobile
///
/// Mirrors the /ws/gesture contract: {"label", "confidence", "timestamp"},
/// debounced/stabilized server-side (or produced on-device in offline mode
/// by TFLiteInferenceService) — never one-per-frame.

class GestureEvent {
  final String label;
  final double confidence;
  final double timestamp;
  final DateTime receivedAt;

  GestureEvent({
    required this.label,
    required this.confidence,
    required this.timestamp,
    DateTime? receivedAt,
  }) : receivedAt = receivedAt ?? DateTime.now();

  factory GestureEvent.fromJson(Map<String, dynamic> json) {
    return GestureEvent(
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      timestamp: (json['timestamp'] as num).toDouble(),
    );
  }
}

/// Mirrors the {"type": "corrected_sentence", ...} event.
class CorrectedSentenceEvent {
  final String sentence;
  final String source; // "gemini" | "flan-t5"
  final bool lowConfidence;
  final DateTime receivedAt;

  CorrectedSentenceEvent({
    required this.sentence,
    required this.source,
    required this.lowConfidence,
    DateTime? receivedAt,
  }) : receivedAt = receivedAt ?? DateTime.now();

  factory CorrectedSentenceEvent.fromJson(Map<String, dynamic> json) {
    return CorrectedSentenceEvent(
      sentence: json['sentence'] as String,
      source: json['source'] as String,
      lowConfidence: json['low_confidence'] as bool? ?? false,
    );
  }
}

/// Mirrors the /ws/speech contract: {"text", "is_final"}.
class SpeechTranscript {
  final String text;
  final bool isFinal;

  SpeechTranscript({required this.text, required this.isFinal});

  factory SpeechTranscript.fromJson(Map<String, dynamic> json) {
    return SpeechTranscript(
      text: json['text'] as String,
      isFinal: json['is_final'] as bool? ?? false,
    );
  }
}
