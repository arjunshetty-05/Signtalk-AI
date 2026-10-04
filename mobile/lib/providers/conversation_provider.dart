/// conversation_provider.dart — SignTalk AI mobile
///
/// Fetches GET /conversations/{user_id} via ApiService. Exposed as a
/// FutureProvider.family so ConversationHistoryScreen's pull-to-refresh can
/// just call ref.refresh().

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/conversation_entry.dart';
import '../services/api_service.dart';

final apiServiceProvider = Provider<ApiService>((ref) => ApiService());

final conversationHistoryProvider =
    FutureProvider.family<List<ConversationEntry>, String>((ref, userId) async {
  final api = ref.read(apiServiceProvider);
  return api.getConversationHistory(userId);
});
