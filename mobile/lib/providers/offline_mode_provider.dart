/// offline_mode_provider.dart — SignTalk AI mobile
///
/// A single Riverpod-managed toggle, read by the networking layer to
/// switch from the live WebSocket path to on-device TFLite inference.
/// Persisted via shared_preferences so it survives app restarts.

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _prefsKey = 'signtalk_offline_mode';

class OfflineModeNotifier extends StateNotifier<bool> {
  OfflineModeNotifier() : super(false) {
    _restore();
  }

  Future<void> _restore() async {
    final prefs = await SharedPreferences.getInstance();
    state = prefs.getBool(_prefsKey) ?? false;
  }

  Future<void> setOffline(bool value) async {
    state = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_prefsKey, value);
  }

  Future<void> toggle() => setOffline(!state);
}

final offlineModeProvider = StateNotifierProvider<OfflineModeNotifier, bool>((ref) {
  return OfflineModeNotifier();
});
