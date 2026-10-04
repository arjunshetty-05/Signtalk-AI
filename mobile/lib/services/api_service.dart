/// api_service.dart — SignTalk AI mobile
///
/// dio-based REST client. Attaches the Firebase Auth JWT as a Bearer header
/// via an interceptor on every call (mirrors the web dashboard's Axios
/// interceptor pattern), so screens never handle tokens manually.

import 'package:dio/dio.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../models/conversation_entry.dart';

class ApiService {
  // NOTE: point this at your deployed backend once live. Defaults to the
  // local dev server for `docker-compose up` / `uvicorn --reload`.
  static const String baseUrl = String.fromEnvironment(
    'SIGNTALK_API_BASE_URL',
    defaultValue: 'http://10.0.2.2:8000', // Android emulator -> host loopback
  );

  final Dio _dio;

  ApiService() : _dio = Dio(BaseOptions(baseUrl: baseUrl)) {
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          final user = FirebaseAuth.instance.currentUser;
          if (user != null) {
            final token = await user.getIdToken();
            options.headers['Authorization'] = 'Bearer $token';
          }
          return handler.next(options);
        },
      ),
    );
  }

  Future<List<ConversationEntry>> getConversationHistory(String userId) async {
    final response = await _dio.get('/conversations/$userId');
    final data = response.data as List<dynamic>;
    return data
        .map((e) => ConversationEntry.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<String> translate(String text, String targetLang, {bool offline = false}) async {
    final response = await _dio.post('/translate', data: {
      'text': text,
      'target_lang': targetLang,
      'offline': offline,
    });
    return response.data['translated_text'] as String;
  }

  Future<List<int>> textToSpeech(String text, String lang, String mode) async {
    final response = await _dio.post<List<int>>(
      '/text-to-speech',
      data: {'text': text, 'lang': lang, 'mode': mode},
      options: Options(responseType: ResponseType.bytes),
    );
    return response.data ?? <int>[];
  }

  Future<Map<String, dynamic>> correctSentence(
    List<String> gestureTokens,
    String emotion,
    List<String> conversationHistory,
  ) async {
    final response = await _dio.post('/ai/predict', data: {
      'gesture_tokens': gestureTokens,
      'emotion': emotion,
      'conversation_history': conversationHistory,
    });
    return response.data as Map<String, dynamic>;
  }
}
