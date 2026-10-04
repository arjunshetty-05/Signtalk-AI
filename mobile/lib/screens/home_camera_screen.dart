/// home_camera_screen.dart — SignTalk AI mobile
///
/// Real-time camera streaming (~15-20fps) over web_socket_channel to
/// /ws/gesture (or on-device TFLite when offline mode is on), with an AI
/// subtitle overlay. Bottom nav routes to history/settings/analytics.

import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image/image.dart' as img;

import '../providers/gesture_provider.dart';
import '../providers/offline_mode_provider.dart';
import '../services/camera_utils.dart';
import '../theme/app_theme.dart';
import '../widgets/live_subtitles.dart';
import 'analytics_screen.dart';
import 'conversation_history_screen.dart';
import 'settings_screen.dart';

const _targetFps = 18;

class HomeCameraScreen extends ConsumerStatefulWidget {
  const HomeCameraScreen({super.key});

  @override
  ConsumerState<HomeCameraScreen> createState() => _HomeCameraScreenState();
}

class _HomeCameraScreenState extends ConsumerState<HomeCameraScreen> {
  CameraController? _controller;
  bool _initialized = false;
  DateTime _lastFrameSentAt = DateTime.fromMillisecondsSinceEpoch(0);
  int _navIndex = 0;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;
    final front = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    final controller = CameraController(front, ResolutionPreset.medium, enableAudio: false);
    await controller.initialize();
    if (!mounted) return;

    controller.startImageStream(_onCameraImage);
    setState(() {
      _controller = controller;
      _initialized = true;
    });
  }

  void _onCameraImage(CameraImage cameraImage) {
    final now = DateTime.now();
    if (now.difference(_lastFrameSentAt).inMilliseconds < (1000 / _targetFps)) return;
    _lastFrameSentAt = now;

    final offline = ref.read(offlineModeProvider);
    final notifier = ref.read(gestureStateProvider.notifier);

    if (offline) {
      notifier.processOfflineFrame(cameraImage);
    } else {
      final rgb = convertCameraImageToRgb(cameraImage);
      final jpegBytes = img.encodeJpg(rgb, quality: 70);
      final base64Jpeg = base64Encode(Uint8List.fromList(jpegBytes));
      notifier.sendFrame(base64Jpeg);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final gestureState = ref.watch(gestureStateProvider);

    return Scaffold(
      body: SafeArea(
        child: IndexedStack(
          index: _navIndex,
          children: [
            _buildCameraTab(gestureState),
            const ConversationHistoryScreen(),
            const AnalyticsScreen(),
            const SettingsScreen(),
          ],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _navIndex,
        onDestinationSelected: (i) => setState(() => _navIndex = i),
        backgroundColor: kBackgroundColor,
        indicatorColor: kNeonColor.withOpacity(0.2),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.camera_alt_outlined), label: 'Camera'),
          NavigationDestination(icon: Icon(Icons.history), label: 'History'),
          NavigationDestination(icon: Icon(Icons.bar_chart_outlined), label: 'Analytics'),
          NavigationDestination(icon: Icon(Icons.settings_outlined), label: 'Settings'),
        ],
      ),
    );
  }

  Widget _buildCameraTab(GestureState gestureState) {
    if (!_initialized || _controller == null) {
      return const Center(child: CircularProgressIndicator(color: kNeonColor));
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(_controller!),

        Positioned(
          top: 16,
          left: 16,
          child: Row(
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  color: gestureState.connected ? kNeonColor : Colors.redAccent,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                gestureState.connected ? 'Live' : 'Reconnecting...',
                style: const TextStyle(color: Colors.white70, fontSize: 12),
              ),
            ],
          ),
        ),

        if (gestureState.latestLabel != null)
          Positioned(
            top: 16,
            right: 16,
            child: GlassPanel(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    gestureState.latestLabel!.label,
                    style: const TextStyle(color: kNeonColor, fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  Text(
                    '${(gestureState.latestLabel!.confidence * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(color: Colors.white54, fontSize: 11),
                  ),
                ],
              ),
            ),
          ),

        Positioned(
          bottom: 24,
          left: 16,
          right: 16,
          child: Center(child: LiveSubtitles(sentenceEvent: gestureState.latestSentence)),
        ),
      ],
    );
  }
}
