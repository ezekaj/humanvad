"""
Brain-Inspired Voice Activity Detection (VAD)
==============================================

Based on neuroscience research from arXiv 2024:
- Temporal gating (two-stage detection like human TVA neurons)
- Harmonic-to-Noise Ratio (voice-specific harmonicity)
- Temporal modulation (syllable-rate 4-8 Hz)
- Hierarchical processing (low + mid-level features)

Target: 85%+ accuracy on realistic noise
Latency: <10ms per frame
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Callable
import time


class BrainInspiredVAD:
    """
    VAD inspired by human auditory cortex processing
    Implements temporal gating and mid-level voice features
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        gating_window_ms: int = 500,  # Sustained response window
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 300,
    ):
        self.sr = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)

        # Temporal gating (mimics human TVA sustained response)
        self.gating_frames = int(gating_window_ms / frame_duration_ms)
        self.detection_history = deque(maxlen=self.gating_frames)  # 500ms history

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
        """Extract low + mid-level features"""

        # 1. Energy (RMS)
        rms = np.sqrt(np.mean(frame ** 2))

        # 2. Spectrum
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), 1/self.sr)

        # 3. Spectral centroid
        if np.sum(spectrum) > 0:
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        else:
            centroid = 0

        # 4. Spectral rolloff
        cumsum = np.cumsum(spectrum)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            rolloff = 0

        # 5. Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        zcr = zero_crossings / (len(frame) / self.sr)

        # 6. Low-frequency ratio
        low_freq_mask = freqs < 1000
        high_freq_mask = freqs >= 1000
        low_energy = np.sum(spectrum[low_freq_mask] ** 2)
        high_energy = np.sum(spectrum[high_freq_mask] ** 2)
        lf_ratio = low_energy / (low_energy + high_energy + 1e-10)

        # 7. Spectral flux
        if self.noise_spectrum is not None and len(spectrum) == len(self.noise_spectrum):
            flux = np.sum((spectrum - self.noise_spectrum) ** 2)
        else:
            flux = 0

        # === NEW: Mid-level features from neuroscience ===

        # 8. Harmonic-to-Noise Ratio (HNR)
        # Voice has strong harmonic structure
        hnr = self._harmonic_to_noise_ratio(frame, spectrum, freqs)

        # 9. Temporal modulation (syllable rate: 4-8 Hz)
        # Voice has characteristic modulation at syllable rate
        modulation = self._temporal_modulation_energy()

        return {
            'rms': rms,
            'centroid': centroid,
            'rolloff': rolloff,
            'flux': flux,
            'zcr': zcr,
            'lf_ratio': lf_ratio,
            'spectrum': spectrum,
            'hnr': hnr,
            'modulation': modulation
        }

    def _harmonic_to_noise_ratio(self, frame: np.ndarray, spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """
        Harmonic-to-Noise Ratio (HNR)

        Voice has strong harmonic structure (fundamental + overtones)
        Noise has flat/random spectrum

        Returns: HNR score [0-1], higher = more harmonic (more voice-like)
        """
        # Find fundamental frequency (F0) via autocorrelation
        # Voice F0 range: 80-400 Hz
        min_lag = int(self.sr / 400)  # Max F0 = 400 Hz
        max_lag = int(self.sr / 80)   # Min F0 = 80 Hz

        if len(frame) < max_lag:
            return 0.0

        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags

        # Find peak in F0 range
        if max_lag < len(autocorr):
            f0_range = autocorr[min_lag:max_lag]
            if len(f0_range) > 0:
                peak_idx = np.argmax(f0_range)
                peak_value = f0_range[peak_idx]

                # Normalize by autocorr at lag 0
                if autocorr[0] > 0:
                    hnr = peak_value / autocorr[0]
                    return min(hnr, 1.0)

        return 0.0

    def _temporal_modulation_energy(self) -> float:
        """
        Temporal modulation at syllable rate (4-8 Hz)

        Voice has characteristic amplitude modulation at syllable rate
        Noise/music has different modulation patterns

        Returns: Modulation energy in syllable-rate band [0-1]
        """
        if len(self.detection_history) < 10:
            return 0.0

        # Get energy envelope from recent frames
        energies = [h['energy'] for h in self.detection_history if 'energy' in h]

        if len(energies) < 10:
            return 0.0

        energies = np.array(energies)

        # Compute modulation spectrum (FFT of energy envelope)
        mod_spectrum = np.abs(np.fft.rfft(energies))
        mod_freqs = np.fft.rfftfreq(len(energies), self.frame_duration_ms / 1000)

        # Energy in syllable-rate band (4-8 Hz)
        syllable_mask = (mod_freqs >= 3) & (mod_freqs <= 9)
        if np.sum(syllable_mask) > 0:
            syllable_energy = np.sum(mod_spectrum[syllable_mask] ** 2)
            total_energy = np.sum(mod_spectrum ** 2)

            if total_energy > 0:
                return min(syllable_energy / total_energy, 1.0)

        return 0.0

    def _classify_speech(self, features: Dict) -> float:
        """
        Classify if features indicate speech

        Returns confidence [0, 1]
        """

        # Adaptive energy threshold (SNR)
        snr = features['rms'] / (self.noise_energy + 1e-6)
        energy_score = min(snr / 2.5, 1.0)

        # Spectral centroid
        centroid = features['centroid']
        if 300 < centroid < 3500:
            centroid_score = 1.0 - abs(centroid - 1000) / 1500
        else:
            centroid_score = 0.0
        centroid_score = max(0, min(centroid_score, 1.0))

        # Spectral rolloff
        rolloff = features['rolloff']
        if 1400 <= rolloff < 6500:
            rolloff_score = 1.0 - abs(rolloff - 3000) / 3000
        else:
            rolloff_score = 0.0
        rolloff_score = max(0, min(rolloff_score, 1.0))

        # Zero-crossing rate
        zcr = features['zcr']
        if 40 < zcr < 250:
            zcr_score = 1.0 - abs(zcr - 120) / 120
        else:
            zcr_score = 0.0
        zcr_score = max(0, min(zcr_score, 1.0))

        # Low-frequency ratio
        lf_ratio = features['lf_ratio']
        if 0.5 < lf_ratio < 0.9:
            lf_score = 1.0 - abs(lf_ratio - 0.7) / 0.3
        else:
            lf_score = 0.0
        lf_score = max(0, min(lf_score, 1.0))

        # Spectral flux
        flux_score = min(features['flux'] / 10.0, 1.0)

        # === NEW: Mid-level features ===

        # Harmonic-to-Noise Ratio (strong discriminator)
        hnr_score = features['hnr']

        # Temporal modulation (syllable rate)
        modulation_score = features['modulation']

        # Weighted combination (emphasize voice-specific features)
        confidence = (
            0.25 * energy_score +      # Energy still important
            0.15 * centroid_score +    # Spectral shape
            0.10 * rolloff_score +
            0.08 * zcr_score +
            0.08 * lf_score +
            0.05 * flux_score +
            0.20 * hnr_score +         # CRITICAL: Harmonicity (voice-specific)
            0.09 * modulation_score    # CRITICAL: Syllable-rate modulation
        )

        return confidence

    def _temporal_gating(self, current_confidence: float) -> bool:
        """
        Two-stage temporal gating (mimics human TVA neurons)

        Stage 1: Onset detection (current frame)
        Stage 2: Sustained response (500ms window)

        Returns: True if both onset detected AND sustained over 500ms
        """
        # Stage 1: Onset detection (stricter threshold)
        onset_detected = current_confidence > 0.40  # Raised from 0.30

        if not onset_detected:
            return False

        # Stage 2: Sustained gating (check last 500ms)
        if len(self.detection_history) < 10:  # Need at least 300ms history (raised from 5)
            return False  # Reject if insufficient history

        # Count how many recent frames detected speech
        recent_detections = sum(1 for h in self.detection_history if h.get('onset', False))

        # Require sustained detection (at least 60% of recent frames - stricter)
        sustained_threshold = len(self.detection_history) * 0.6  # Raised from 0.4
        sustained_detected = recent_detections >= sustained_threshold

        return onset_detected and sustained_detected

    def _update_noise_model(self, features: Dict, is_speech: bool):
        """Update noise model during silence"""
        if not is_speech:
            alpha = 0.1
            self.noise_energy = (1 - alpha) * self.noise_energy + alpha * features['rms']

            if self.noise_spectrum is None:
                self.noise_spectrum = features['spectrum'].copy()
            else:
                self.noise_spectrum = (1 - alpha) * self.noise_spectrum + alpha * features['spectrum']

            self.noise_update_count += 1

    def detect_frame(self, frame: np.ndarray) -> Dict:
        """
        Detect speech in single frame using brain-inspired mechanisms

        Returns:
            Dict with is_speech, confidence, latency_ms
        """
        start_time = time.perf_counter()

        # Ensure correct frame size
        if len(frame) != self.frame_samples:
            frame = np.pad(frame, (0, max(0, self.frame_samples - len(frame))))[:self.frame_samples]

        # Extract features
        features = self._compute_features(frame)

        # Classify (onset detection)
        confidence = self._classify_speech(features)

        # Store in history for temporal processing
        onset_detected = confidence > 0.30
        self.detection_history.append({
            'confidence': confidence,
            'onset': onset_detected,
            'energy': features['rms']
        })

        # Temporal gating (sustained response)
        is_speech = self._temporal_gating(confidence)

        # Update noise model
        self._update_noise_model(features, is_speech)

        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(latency_ms)

        return {
            'is_speech': is_speech,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'features': features,
            'onset': onset_detected
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
        self.detection_history.clear()


def demo():
    """Quick demo"""
    print("=" * 80)
    print(" BRAIN-INSPIRED VAD DEMO")
    print("=" * 80)
    print()
    print("Features:")
    print("  - Temporal gating (two-stage detection like human TVA)")
    print("  - Harmonic-to-Noise Ratio (voice harmonicity)")
    print("  - Temporal modulation (syllable-rate 4-8 Hz)")
    print("  - Hierarchical processing (low + mid-level)")
    print()

    sr = 16000
    vad = BrainInspiredVAD(sample_rate=sr)

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
    print("Run test_realistic_noise.py with BrainInspiredVAD for full evaluation")


if __name__ == "__main__":
    demo()
