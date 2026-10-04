/// conversation_entry.dart — SignTalk AI mobile
///
/// Mirrors one entry from GET /conversations/{user_id}.

class ConversationEntry {
  final String sentence;
  final String language;
  final String emotion;
  final double createdAt;

  ConversationEntry({
    required this.sentence,
    required this.language,
    required this.emotion,
    required this.createdAt,
  });

  factory ConversationEntry.fromJson(Map<String, dynamic> json) {
    return ConversationEntry(
      sentence: json['sentence'] as String? ?? '',
      language: json['language'] as String? ?? 'en',
      emotion: json['emotion'] as String? ?? 'neutral',
      createdAt: (json['created_at'] as num?)?.toDouble() ?? 0.0,
    );
  }

  DateTime get createdAtDateTime =>
      DateTime.fromMillisecondsSinceEpoch((createdAt * 1000).round());
}
