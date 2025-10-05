"""
Production-Ready Voice Activity Detection (VAD)
===============================================

Optimized for:
- Person speaking vs background noise
- Real-world noise (babble, music, traffic, TV)
- Ultra-fast (<10ms per frame)
- High accuracy (>85% in noise)

Strategy:
- Adaptive energy threshold (handles varying noise levels)
- Spectral features (distinguish speech from noise spectrum)
- Minimal temporal dependency (works on short frames)
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Callable
import time


class ProductionVAD:
    """
    Production VAD optimized for distinguishing human speech from background noise
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,  # 30ms frames for better features
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 300,
    ):
        self.sr = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)

        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)

        # Adaptive noise tracking
        self.noise_energy = 0.01
        self.noise_spectrum = None
        self.noise_update_count = 0

        # State
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0

        # Performance
        self.processing_times = deque(maxlen=100)

    def _compute_features(self, frame: np.ndarray) -> Dict:
        """Extract speech-discriminative features"""

        # 1. Energy (RMS)
        rms = np.sqrt(np.mean(frame ** 2))

        # 2. Spectrum
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), 1/self.sr)

        # 3. Spectral centroid (speech: 500-2500 Hz)
        if np.sum(spectrum) > 0:
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        else:
            centroid = 0

        # 4. Spectral rolloff (95% energy point)
        cumsum = np.cumsum(spectrum)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            rolloff = 0

        # 5. Spectral flux (change in spectrum)
        if self.noise_spectrum is not None and len(spectrum) == len(self.noise_spectrum):
            flux = np.sum((spectrum - self.noise_spectrum) ** 2)
        else:
            flux = 0

        # 6. Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        zcr = zero_crossings / (len(frame) / self.sr)

        # 7. Low-frequency energy ratio (speech has more low-freq energy)
        low_freq_mask = freqs < 1000
        high_freq_mask = freqs >= 1000
        low_energy = np.sum(spectrum[low_freq_mask] ** 2)
        high_energy = np.sum(spectrum[high_freq_mask] ** 2)
        lf_ratio = low_energy / (low_energy + high_energy + 1e-10)

        return {
            'rms': rms,
            'centroid': centroid,
            'rolloff': rolloff,
            'flux': flux,
            'zcr': zcr,
            'lf_ratio': lf_ratio,
            'spectrum': spectrum
        }

    def _classify_speech(self, features: Dict) -> float:
        """
        Classify if features indicate speech

        Returns confidence [0, 1]
        """

        # Adaptive energy threshold (SNR)
        snr = features['rms'] / (self.noise_energy + 1e-6)
        energy_score = min(snr / 2.5, 1.0)  # More sensitive to energy

        # Spectral centroid (speech: 300-3000 Hz - wider range)
        centroid = features['centroid']
        if 300 < centroid < 3500:
            centroid_score = 1.0 - abs(centroid - 1000) / 1500
        else:
            centroid_score = 0.0
        centroid_score = max(0, min(centroid_score, 1.0))

        # Spectral rolloff (speech: 1400-6000 Hz - inclusive boundary)
        rolloff = features['rolloff']
        if 1400 <= rolloff < 6500:
            rolloff_score = 1.0 - abs(rolloff - 3000) / 3000
        else:
            rolloff_score = 0.0
        rolloff_score = max(0, min(rolloff_score, 1.0))

        # Zero-crossing rate (speech: 50-200 Hz)
        zcr = features['zcr']
        if 40 < zcr < 250:
            zcr_score = 1.0 - abs(zcr - 120) / 120
        else:
            zcr_score = 0.0
        zcr_score = max(0, min(zcr_score, 1.0))

        # Low-frequency ratio (speech has 60-80% low-freq energy)
        lf_ratio = features['lf_ratio']
        if 0.5 < lf_ratio < 0.9:
            lf_score = 1.0 - abs(lf_ratio - 0.7) / 0.3
        else:
            lf_score = 0.0
        lf_score = max(0, min(lf_score, 1.0))

        # Spectral flux (speech has dynamic spectrum)
        flux_score = min(features['flux'] / 10.0, 1.0)

        # Weighted combination (no strict gating - let features vote)
        confidence = (
            0.35 * energy_score +      # High energy needed
            0.20 * centroid_score +    # Centroid in speech range
            0.15 * rolloff_score +     # Rolloff in speech range
            0.12 * zcr_score +         # ZCR in speech range
            0.10 * lf_score +          # Low-freq dominance
            0.08 * flux_score          # Dynamic spectrum
        )

        return confidence

    def _update_noise_model(self, features: Dict, is_speech: bool):
        """Update noise model during silence"""
        if not is_speech:
            # Exponential moving average
            alpha = 0.1
            self.noise_energy = (1 - alpha) * self.noise_energy + alpha * features['rms']

            if self.noise_spectrum is None:
                self.noise_spectrum = features['spectrum'].copy()
            else:
                self.noise_spectrum = (1 - alpha) * self.noise_spectrum + alpha * features['spectrum']

            self.noise_update_count += 1

    def detect_frame(self, frame: np.ndarray) -> Dict:
        """
        Detect speech in single frame

        Returns:
            Dict with is_speech, confidence, latency_ms
        """
        start_time = time.perf_counter()

        # Ensure correct frame size
        if len(frame) != self.frame_samples:
            frame = np.pad(frame, (0, max(0, self.frame_samples - len(frame))))[:self.frame_samples]

        # Extract features
        features = self._compute_features(frame)

        # Classify
        confidence = self._classify_speech(features)

        # Decision with threshold
        threshold = 0.35  # Lower threshold for better recall in noise
        is_speech = confidence > threshold

        # Update noise model
        self._update_noise_model(features, is_speech)

        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(latency_ms)

        return {
            'is_speech': is_speech,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'features': features
        }

    def process_stream(
        self,
        frame: np.ndarray,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None
    ) -> Dict:
        """
        Process streaming audio with state tracking
        """
        result = self.detect_frame(frame)

        if result['is_speech']:
            self.speech_frame_count += 1
            self.silence_frame_count = 0

            if not self.is_speaking and self.speech_frame_count >= self.min_speech_frames:
                self.is_speaking = True
                if on_speech_start:
                    on_speech_start()
        else:
            self.silence_frame_count += 1

            if self.is_speaking and self.silence_frame_count >= self.min_silence_frames:
                self.is_speaking = False
                self.speech_frame_count = 0
                if on_speech_end:
                    on_speech_end()

        return {
            'is_speaking': self.is_speaking,
            'confidence': result['confidence'],
            'latency_ms': result['latency_ms'],
            'speech_frames': self.speech_frame_count,
            'silence_frames': self.silence_frame_count
        }

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if len(self.processing_times) > 0:
            return {
                'avg_latency_ms': np.mean(self.processing_times),
                'p50_latency_ms': np.percentile(self.processing_times, 50),
                'p95_latency_ms': np.percentile(self.processing_times, 95),
                'max_latency_ms': np.max(self.processing_times),
                'noise_energy': self.noise_energy
            }
        return {}

    def reset(self):
        """Reset state"""
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0


def demo():
    """Quick demo"""
    print("=" * 80)
    print(" PRODUCTION VAD DEMO")
    print("=" * 80)
    print()

    sr = 16000
    vad = ProductionVAD(sample_rate=sr)

    # Test speed
    print("Speed Test:")
    print("-" * 80)

    test_frame = np.random.randn(480)  # 30ms frame
    n_iterations = 100

    start = time.perf_counter()
    for _ in range(n_iterations):
        vad.detect_frame(test_frame)
    avg_time = (time.perf_counter() - start) / n_iterations * 1000

    print(f"Average latency: {avg_time:.2f}ms per frame (30ms audio)")
    print(f"Real-time factor: {avg_time / 30:.2f}x")
    print(f"Target: <10ms {'PASS' if avg_time < 10 else 'FAIL'}")
    print()

    stats = vad.get_stats()
    print(f"P50: {stats['p50_latency_ms']:.2f}ms")
    print(f"P95: {stats['p95_latency_ms']:.2f}ms")
    print()

    print("=" * 80)
    print()
    print("Run test_realistic_noise.py with ProductionVAD for full evaluation")


if __name__ == "__main__":
    demo()
