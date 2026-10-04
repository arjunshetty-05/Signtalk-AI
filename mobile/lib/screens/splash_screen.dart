/// splash_screen.dart — SignTalk AI mobile
///
/// App branding + checks auth state, then routes to auth or home.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/auth_provider.dart';
import '../theme/app_theme.dart';
import 'auth_screen.dart';
import 'home_camera_screen.dart';

class SplashScreen extends ConsumerWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authStateProvider);

    return authState.when(
      loading: () => const _SplashBranding(),
      error: (_, __) => const AuthScreen(),
      data: (user) => user == null ? const AuthScreen() : const HomeCameraScreen(),
    );
  }
}

class _SplashBranding extends StatelessWidget {
  const _SplashBranding();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.sign_language, size: 64, color: kNeonColor),
            const SizedBox(height: 16),
            Text(
              'SignTalk AI',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: kNeonColor,
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 24),
            const CircularProgressIndicator(color: kNeonColor),
          ],
        ),
      ),
    );
  }
}
