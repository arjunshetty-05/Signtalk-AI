/// settings_screen.dart — SignTalk AI mobile
///
/// Language preference (EN/HI/KN), offline mode toggle, logout, account info.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/auth_provider.dart';
import '../providers/language_provider.dart';
import '../providers/offline_mode_provider.dart';
import '../theme/app_theme.dart';
import 'offline_mode_screen.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final user = ref.watch(authStateProvider).value;
    final language = ref.watch(languageProvider);
    final isOffline = ref.watch(offlineModeProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          GlassPanel(
            child: Row(
              children: [
                const Icon(Icons.account_circle_outlined, color: kNeonColor, size: 32),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    user?.email ?? 'Unknown user',
                    style: const TextStyle(color: Colors.white, fontSize: 14),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          GlassPanel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Language', style: TextStyle(color: Colors.white70, fontSize: 12)),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  children: supportedLanguages.entries.map((entry) {
                    final selected = entry.key == language;
                    return ChoiceChip(
                      label: Text(entry.value),
                      selected: selected,
                      selectedColor: kNeonColor.withOpacity(0.25),
                      labelStyle: TextStyle(color: selected ? kNeonColor : Colors.white70),
                      onSelected: (_) => ref.read(languageProvider.notifier).setLanguage(entry.key),
                    );
                  }).toList(),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          GlassPanel(
            child: ListTile(
              contentPadding: EdgeInsets.zero,
              leading: Icon(isOffline ? Icons.airplanemode_active : Icons.wifi, color: kNeonColor),
              title: const Text('Offline mode', style: TextStyle(color: Colors.white)),
              subtitle: Text(isOffline ? 'On — on-device inference' : 'Off — live backend',
                  style: const TextStyle(color: Colors.white38, fontSize: 11)),
              trailing: Switch(
                value: isOffline,
                activeColor: kNeonColor,
                onChanged: (value) => ref.read(offlineModeProvider.notifier).setOffline(value),
              ),
              onTap: () => Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const OfflineModeScreen()),
              ),
            ),
          ),
          const SizedBox(height: 16),

          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () => ref.read(authControllerProvider).logout(),
              icon: const Icon(Icons.logout, color: Colors.redAccent),
              label: const Text('Log out', style: TextStyle(color: Colors.redAccent)),
              style: OutlinedButton.styleFrom(side: const BorderSide(color: Colors.redAccent)),
            ),
          ),
        ],
      ),
    );
  }
}
