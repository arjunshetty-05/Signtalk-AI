/// live_subtitles.dart — SignTalk AI mobile
///
/// Reusable captions widget: dark, high-contrast styling for outdoor
/// visibility. Renders discrete (sparse, debounced) events with a
/// hold-then-fade behavior — it only re-triggers its animation when a
/// genuinely new sentence arrives (deduped by text + receivedAt), never on
/// a rebuild/frame tick.

import 'dart:async';

import 'package:flutter/material.dart';

import '../models/gesture_event.dart';
import '../theme/app_theme.dart';

const _holdDuration = Duration(seconds: 4);

class LiveSubtitles extends StatefulWidget {
  final CorrectedSentenceEvent? sentenceEvent;

  const LiveSubtitles({super.key, this.sentenceEvent});

  @override
  State<LiveSubtitles> createState() => _LiveSubtitlesState();
}

class _LiveSubtitlesState extends State<LiveSubtitles> {
  CorrectedSentenceEvent? _displayed;
  DateTime? _lastKey;
  Timer? _fadeTimer;

  @override
  void didUpdateWidget(covariant LiveSubtitles oldWidget) {
    super.didUpdateWidget(oldWidget);
    final event = widget.sentenceEvent;
    if (event == null) return;
    if (_lastKey == event.receivedAt) return; // dedupe — not a new event

    _lastKey = event.receivedAt;
    setState(() => _displayed = event);

    _fadeTimer?.cancel();
    _fadeTimer = Timer(_holdDuration, () {
      if (mounted) setState(() => _displayed = null);
    });
  }

  @override
  void dispose() {
    _fadeTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 200),
      transitionBuilder: (child, animation) => FadeTransition(
        opacity: animation,
        child: SlideTransition(
          position: Tween<Offset>(begin: const Offset(0, 0.15), end: Offset.zero).animate(animation),
          child: child,
        ),
      ),
      child: _displayed == null
          ? const SizedBox.shrink(key: ValueKey('empty'))
          : GlassPanel(
              key: ValueKey('${_displayed!.sentence}-${_displayed!.receivedAt}'),
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    _displayed!.sentence,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        _displayed!.source == 'gemini' ? 'Gemini 2.0 Flash' : 'Flan-T5 (offline)',
                        style: const TextStyle(color: Colors.white54, fontSize: 11),
                      ),
                      if (_displayed!.lowConfidence) ...[
                        const SizedBox(width: 8),
                        const Text('Low confidence',
                            style: TextStyle(color: Colors.amber, fontSize: 11, fontWeight: FontWeight.bold)),
                      ],
                    ],
                  ),
                ],
              ),
            ),
    );
  }
}
