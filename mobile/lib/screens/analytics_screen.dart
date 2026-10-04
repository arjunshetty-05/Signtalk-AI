/// analytics_screen.dart — SignTalk AI mobile
///
/// fl_chart charts: session prediction confidence over time (a rolling
/// buffer of the sparse /ws/gesture events — not per-frame samples) and
/// emotion distribution (sourced from conversation history entries, same
/// approach as the web dashboard's AnalyticsPanel).

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/auth_provider.dart';
import '../providers/conversation_provider.dart';
import '../providers/gesture_provider.dart';
import '../theme/app_theme.dart';

const _maxConfidencePoints = 30;

class AnalyticsScreen extends ConsumerStatefulWidget {
  const AnalyticsScreen({super.key});

  @override
  ConsumerState<AnalyticsScreen> createState() => _AnalyticsScreenState();
}

class _AnalyticsScreenState extends ConsumerState<AnalyticsScreen> {
  final List<double> _confidenceHistory = [];
  double? _lastTimestamp;

  @override
  Widget build(BuildContext context) {
    final gestureState = ref.watch(gestureStateProvider);
    final label = gestureState.latestLabel;

    if (label != null && label.timestamp != _lastTimestamp) {
      _lastTimestamp = label.timestamp;
      _confidenceHistory.add(label.confidence);
      if (_confidenceHistory.length > _maxConfidencePoints) {
        _confidenceHistory.removeAt(0);
      }
    }

    final userId = ref.watch(authStateProvider).value?.uid;
    final historyAsync = userId == null
        ? null
        : ref.watch(conversationHistoryProvider(userId));

    final emotionCounts = <String, int>{};
    historyAsync?.whenData((entries) {
      for (final entry in entries) {
        emotionCounts[entry.emotion] = (emotionCounts[entry.emotion] ?? 0) + 1;
      }
    });

    return Scaffold(
      appBar: AppBar(title: const Text('Analytics')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          GlassPanel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Confidence over time', style: TextStyle(color: Colors.white70, fontSize: 12)),
                const SizedBox(height: 12),
                SizedBox(
                  height: 140,
                  child: _confidenceHistory.isEmpty
                      ? const Center(
                          child: Text('No predictions yet.', style: TextStyle(color: Colors.white38, fontSize: 12)))
                      : LineChart(
                          LineChartData(
                            minY: 0,
                            maxY: 1,
                            gridData: const FlGridData(show: false),
                            titlesData: const FlTitlesData(show: false),
                            borderData: FlBorderData(show: false),
                            lineBarsData: [
                              LineChartBarData(
                                spots: [
                                  for (var i = 0; i < _confidenceHistory.length; i++)
                                    FlSpot(i.toDouble(), _confidenceHistory[i]),
                                ],
                                isCurved: true,
                                color: kNeonColor,
                                barWidth: 2,
                                dotData: const FlDotData(show: false),
                              ),
                            ],
                          ),
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
                const Text('Emotion distribution', style: TextStyle(color: Colors.white70, fontSize: 12)),
                const SizedBox(height: 12),
                SizedBox(
                  height: 160,
                  child: emotionCounts.isEmpty
                      ? const Center(
                          child: Text('No data yet.', style: TextStyle(color: Colors.white38, fontSize: 12)))
                      : BarChart(
                          BarChartData(
                            gridData: const FlGridData(show: false),
                            borderData: FlBorderData(show: false),
                            titlesData: FlTitlesData(
                              leftTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                              topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                              rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                              bottomTitles: AxisTitles(
                                sideTitles: SideTitles(
                                  showTitles: true,
                                  getTitlesWidget: (value, meta) {
                                    final keys = emotionCounts.keys.toList();
                                    final idx = value.toInt();
                                    if (idx < 0 || idx >= keys.length) return const SizedBox.shrink();
                                    return Text(keys[idx],
                                        style: const TextStyle(color: Colors.white38, fontSize: 9));
                                  },
                                ),
                              ),
                            ),
                            barGroups: [
                              for (var i = 0; i < emotionCounts.length; i++)
                                BarChartGroupData(x: i, barRods: [
                                  BarChartRodData(
                                    toY: emotionCounts.values.elementAt(i).toDouble(),
                                    color: kNeonColor,
                                    width: 16,
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                ]),
                            ],
                          ),
                        ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
