/// auth_provider.dart — SignTalk AI mobile
///
/// Riverpod-managed Firebase Auth state + login/signup/logout actions.
/// No setState-based global state anywhere in this app — Riverpod only.

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final authStateProvider = StreamProvider<User?>((ref) {
  return FirebaseAuth.instance.authStateChanges();
});

final authControllerProvider = Provider<AuthController>((ref) {
  return AuthController();
});

class AuthController {
  final _auth = FirebaseAuth.instance;

  Future<void> login(String email, String password) {
    return _auth.signInWithEmailAndPassword(email: email, password: password);
  }

  Future<void> signup(String email, String password) {
    return _auth.createUserWithEmailAndPassword(email: email, password: password);
  }

  Future<void> logout() => _auth.signOut();

  Future<String?> currentToken() => _auth.currentUser?.getIdToken();
}
