/// offline_mode_screen.dart — SignTalk AI mobile
///
/// Dedicated offline-mode explainer + toggle (also surfaced compactly in
/// SettingsScreen). Flipping this switches gesture recognition from the
/// live /ws/gesture path to on-device TFLite inference, and routes
/// translation through the bundled offline_phrasebook.json asset.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/offline_mode_provider.dart';
import '../theme/app_theme.dart';

class OfflineModeScreen extends ConsumerWidget {
  const OfflineModeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isOffline = ref.watch(offlineModeProvider);
    final notifier = ref.read(offlineModeProvider.notifier);

    return Scaffold(
      appBar: AppBar(title: const Text('Offline Mode')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: GlassPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(isOffline ? Icons.airplanemode_active : Icons.wifi, color: kNeonColor),
                  const SizedBox(width: 12),
                  const Expanded(
                    child: Text('Offline mode', style: TextStyle(color: Colors.white, fontSize: 16)),
                  ),
                  Switch(
                    value: isOffline,
                    activeColor: kNeonColor,
                    onChanged: (value) => notifier.setOffline(value),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              const Text(
                'When on: gesture recognition runs fully on-device via '
                'movenet.tflite + bilstm.tflite, sentence correction skips '
                'Gemini for the local Flan-T5 fallback, translation only '
                'checks the bundled phrasebook, and text-to-speech uses '
                'Coqui instead of gTTS. No network access is required.',
                style: TextStyle(color: Colors.white54, fontSize: 12, height: 1.4),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
