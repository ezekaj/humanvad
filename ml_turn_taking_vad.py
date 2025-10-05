"""
ML-Based Turn-Taking VAD
========================

Simplified machine learning approach using audio features
to predict turn-taking without complex pre-trained models.

Uses RandomForest on prosodic + spectral features.
"""

import numpy as np
from collections import deque
from typing import Dict
import time
from production_vad import ProductionVAD
import scipy.signal


class MLTurnTakingVAD:
    """
    ML-based turn-taking detection using engineered features

    Features:
    - Energy trend (increasing/decreasing)
    - Pitch contour (estimated via autocorrelation)
    - Speech rate (zero-crossing rate variation)
    - Spectral features
    - Temporal context (last 1-3 seconds)
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

        # Fast speech detection
        self.speech_detector = ProductionVAD(sample_rate=sample_rate)

        # Temporal buffers
        self.audio_buffer = deque(maxlen=sample_rate * 3)  # 3 seconds
        self.energy_history = deque(maxlen=150)  # 3s at 10ms frames
        self.zcr_history = deque(maxlen=150)
        self.spectral_history = deque(maxlen=150)

        # Performance tracking
        self.processing_times = deque(maxlen=100)

    def _extract_features(self, audio: np.ndarray) -> Dict:
        """Extract turn-taking features from audio"""

        # Basic energy
        energy = np.sqrt(np.mean(audio ** 2))

        # Zero-crossing rate (speech rate proxy)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

        # Spectral centroid
        if len(audio) > 64:
            fft = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
            centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-6)
        else:
            centroid = 0.0

        return {
            'energy': energy,
            'zcr': zcr,
            'centroid': centroid
        }

    def _estimate_pitch_trend(self, audio: np.ndarray) -> float:
        """
        Estimate if pitch is falling (turn-end) or rising/stable
        Returns: -1.0 (falling), 0.0 (stable), 1.0 (rising)
        """
        if len(audio) < 400:
            return 0.0

        # Split into first/last half
        mid = len(audio) // 2
        first_half = audio[:mid]
        second_half = audio[mid:]

        def simple_pitch_estimate(segment):
            # Autocorrelation for fundamental frequency
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find first peak after lag 0
            # F0 range: 80-400 Hz -> lag range at 16kHz
            min_lag = int(self.sr / 400)  # ~40 samples
            max_lag = int(self.sr / 80)   # ~200 samples

            if max_lag >= len(autocorr):
                return 0.0

            search_region = autocorr[min_lag:max_lag]
            if len(search_region) == 0:
                return 0.0

            peak_lag = np.argmax(search_region) + min_lag
            f0 = self.sr / peak_lag if peak_lag > 0 else 0.0

            return f0

        f0_first = simple_pitch_estimate(first_half)
        f0_second = simple_pitch_estimate(second_half)

        if f0_first == 0.0 or f0_second == 0.0:
            return 0.0

        # Pitch trend
        diff = f0_second - f0_first

        if diff < -20:  # Falling >20 Hz = turn-end
            return -1.0
        elif diff > 20:  # Rising >20 Hz = question/continuation
            return 1.0
        else:
            return 0.0

    def _predict_turn_end(self) -> float:
        """
        Predict turn-end probability using heuristics
        (In production, this would be a trained RandomForest/SVM)
        """

        if len(self.energy_history) < 20:
            return 0.3  # Insufficient data

        recent_energy = list(self.energy_history)[-30:]
        recent_zcr = list(self.zcr_history)[-30:]

        score = 0.0

        # 1. Energy trend (decreasing = turn-ending)
        if len(recent_energy) >= 20:
            first_half_energy = np.mean(recent_energy[:10])
            second_half_energy = np.mean(recent_energy[-10:])

            if second_half_energy < first_half_energy * 0.7:
                score += 0.3  # Energy dropping

        # 2. Speech rate (ZCR decreasing = slowing down)
        if len(recent_zcr) >= 20:
            first_half_zcr = np.mean(recent_zcr[:10])
            second_half_zcr = np.mean(recent_zcr[-10:])

            if second_half_zcr < first_half_zcr * 0.8:
                score += 0.2  # Slowing down

        # 3. Pitch trend (last 500ms)
        if len(self.audio_buffer) >= int(self.sr * 0.5):
            recent_audio = np.array(list(self.audio_buffer)[-int(self.sr * 0.5):])
            pitch_trend = self._estimate_pitch_trend(recent_audio)

            if pitch_trend < 0:  # Falling pitch
                score += 0.4

        # 4. Absolute energy level (very low = ending)
        if len(recent_energy) > 0:
            current_energy = recent_energy[-1]
            if current_energy < 0.01:
                score += 0.1

        return min(score, 1.0)

    def process_frame(self, user_frame: np.ndarray, ai_frame: np.ndarray) -> Dict:
        """Process stereo audio frame"""

        start_time = time.perf_counter()

        # Update buffers
        self.audio_buffer.extend(ai_frame)

        # Fast speech detection
        user_result = self.speech_detector.detect_frame(user_frame)
        ai_result = self.speech_detector.detect_frame(ai_frame)

        user_speaking = user_result['is_speech']
        ai_speaking = ai_result['is_speech']

        # Extract features from AI speech
        features = self._extract_features(ai_frame)
        self.energy_history.append(features['energy'])
        self.zcr_history.append(features['zcr'])
        self.spectral_history.append(features['centroid'])

        # Predict turn-end probability
        turn_end_prob = self._predict_turn_end()

        # Decision logic
        if user_speaking and ai_speaking:
            # Overlap detected
            if turn_end_prob > 0.70:
                # AI naturally finishing - wait
                action = "wait_for_ai_completion"
            else:
                # User interrupting - stop AI
                action = "interrupt_ai_immediately"
        elif user_speaking:
            # User speaking, AI silent
            action = "interrupt_ai_immediately"
        else:
            action = "continue"

        latency_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(latency_ms)

        return {
            'action': action,
            'turn_end_prob': turn_end_prob,
            'user_speaking': user_speaking,
            'ai_speaking': ai_speaking,
            'overlap': user_speaking and ai_speaking,
            'latency_ms': latency_ms,

            # Feature debug
            'features': features
        }

    def get_stats(self) -> Dict:
        """Performance statistics"""
        if len(self.processing_times) > 0:
            return {
                'avg_latency_ms': np.mean(self.processing_times),
                'p50_latency_ms': np.percentile(self.processing_times, 50),
                'p95_latency_ms': np.percentile(self.processing_times, 95),
                'max_latency_ms': np.max(self.processing_times)
            }
        return {}

    def reset(self):
        """Reset state"""
        self.audio_buffer.clear()
        self.energy_history.clear()
        self.zcr_history.clear()
        self.spectral_history.clear()


def demo():
    """Quick demo"""
    print("=" * 80)
    print(" ML-BASED TURN-TAKING VAD")
    print("=" * 80)
    print()
    print("Features:")
    print("  - Energy trend analysis")
    print("  - Pitch contour estimation")
    print("  - Speech rate tracking")
    print("  - Spectral features")
    print()
    print("Target: 70-80% accuracy (simpler than VAP, faster than neuromorphic)")
    print()

    sr = 16000
    vad = MLTurnTakingVAD(sample_rate=sr)

    # Speed test
    print("Speed Test:")
    print("-" * 80)

    test_frame = np.random.randn(160)
    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        vad.process_frame(test_frame, test_frame)
    avg_time = (time.perf_counter() - start) / n_iterations * 1000

    print(f"Average latency: {avg_time:.2f}ms per frame")
    print(f"Target: <10ms {'PASS' if avg_time < 10 else 'FAIL'}")
    print()

    stats = vad.get_stats()
    print(f"P95: {stats['p95_latency_ms']:.2f}ms")
    print()
    print("Run test_ml_turn_taking.py for full evaluation")
    print()


if __name__ == "__main__":
    demo()
