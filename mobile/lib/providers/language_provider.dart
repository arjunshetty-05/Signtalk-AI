/// language_provider.dart — SignTalk AI mobile
///
/// EN/HI/KN language preference, persisted via shared_preferences.

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _prefsKey = 'signtalk_language';
const supportedLanguages = {'en': 'English', 'hi': 'हिन्दी', 'kn': 'ಕನ್ನಡ'};

class LanguageNotifier extends StateNotifier<String> {
  LanguageNotifier() : super('en') {
    _restore();
  }

  Future<void> _restore() async {
    final prefs = await SharedPreferences.getInstance();
    state = prefs.getString(_prefsKey) ?? 'en';
  }

  Future<void> setLanguage(String langCode) async {
    state = langCode;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsKey, langCode);
  }
}

final languageProvider = StateNotifierProvider<LanguageNotifier, String>((ref) {
  return LanguageNotifier();
});
