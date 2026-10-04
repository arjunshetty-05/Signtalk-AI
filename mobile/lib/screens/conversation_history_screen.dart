/// conversation_history_screen.dart — SignTalk AI mobile
///
/// Fetches GET /conversations/{user_id} via ApiService (dio + Firebase
/// JWT), reverse-chronological list with language + emotion tags,
/// pull-to-refresh.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/auth_provider.dart';
import '../providers/conversation_provider.dart';
import '../theme/app_theme.dart';

const _emotionColors = {
  'happy': Colors.amber,
  'sad': Colors.blueAccent,
  'angry': Colors.redAccent,
  'fear': Colors.deepPurpleAccent,
  'surprise': Colors.pinkAccent,
  'neutral': Colors.grey,
  'disgust': Colors.greenAccent,
};

class ConversationHistoryScreen extends ConsumerWidget {
  const ConversationHistoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authStateProvider);
    final userId = authState.value?.uid;

    if (userId == null) {
      return const Center(child: Text('Not signed in', style: TextStyle(color: Colors.white54)));
    }

    final historyAsync = ref.watch(conversationHistoryProvider(userId));

    return Scaffold(
      backgroundColor: Colors.transparent,
      appBar: AppBar(title: const Text('History')),
      body: RefreshIndicator(
        color: kNeonColor,
        onRefresh: () async => ref.refresh(conversationHistoryProvider(userId).future),
        child: historyAsync.when(
          loading: () => const Center(child: CircularProgressIndicator(color: kNeonColor)),
          error: (err, _) => ListView(
            children: const [
              Padding(
                padding: EdgeInsets.all(24),
                child: Text('Could not load conversation history.', style: TextStyle(color: Colors.redAccent)),
              ),
            ],
          ),
          data: (entries) {
            if (entries.isEmpty) {
              return ListView(
                children: const [
                  Padding(
                    padding: EdgeInsets.all(24),
                    child: Text('No conversations yet.', style: TextStyle(color: Colors.white54)),
                  ),
                ],
              );
            }
            return ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: entries.length,
              itemBuilder: (context, index) {
                final entry = entries[index];
                return Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: GlassPanel(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(entry.sentence, style: const TextStyle(color: Colors.white, fontSize: 14)),
                        const SizedBox(height: 6),
                        Row(
                          children: [
                            Text(entry.language.toUpperCase(),
                                style: const TextStyle(color: Colors.white38, fontSize: 10)),
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                              decoration: BoxDecoration(
                                color: (_emotionColors[entry.emotion] ?? Colors.grey).withOpacity(0.2),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Text(
                                entry.emotion,
                                style: TextStyle(
                                  color: _emotionColors[entry.emotion] ?? Colors.grey,
                                  fontSize: 10,
                                ),
                              ),
                            ),
                            const Spacer(),
                            Text(
                              '${entry.createdAtDateTime.hour.toString().padLeft(2, '0')}:${entry.createdAtDateTime.minute.toString().padLeft(2, '0')}',
                              style: const TextStyle(color: Colors.white24, fontSize: 10),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                );
              },
            );
          },
        ),
      ),
    );
  }
}
