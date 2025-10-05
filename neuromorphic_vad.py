"""
Neuromorphic VAD - Brain-Inspired Turn-Taking Detection
========================================================

Implements actual human brain mechanisms for audio-only interruption detection:
1. Theta oscillations (4-8 Hz) - 200ms temporal segmentation
2. STG onset detection - Boundaries after >=200ms silence
3. Cortical entrainment - Rhythm tracking and prediction
4. Hierarchical predictive coding - Multi-timescale context

Target: 95-100% accuracy using only audio (matches human performance)
"""

import numpy as np
from collections import deque
from typing import Dict, Optional
import time
import scipy.signal
from production_vad import ProductionVAD


class ThetaOscillator:
    """
    Simulates theta-band (4-8 Hz) oscillations in auditory cortex

    Research: Theta phase segments speech into ~200ms windows
    and predicts timing of acoustic boundaries
    """

    def __init__(self, sample_rate: int = 16000, center_freq: float = 5.0):
        self.sr = sample_rate
        self.freq = center_freq  # 5 Hz = 200ms period
        self.phase = 0.0
        self.time = 0.0

    def get_phase(self) -> float:
        """Current phase (0-2*pi)"""
        return self.phase % (2 * np.pi)

    def update(self, dt: float):
        """Advance oscillator by time step"""
        self.phase += 2 * np.pi * self.freq * dt
        self.time += dt

    def set_frequency(self, new_freq: float):
        """Adjust frequency to entrain to speech rhythm"""
        self.freq = np.clip(new_freq, 4.0, 8.0)  # Keep in theta band


class STG_OnsetDetector:
    """
    Superior Temporal Gyrus onset detection

    Research: Posterior STG responds to onsets after >=200ms silence
    These mark phrase/turn boundaries
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.silence_threshold_ms = 200.0  # Critical boundary marker
        self.current_silence_ms = 0.0
        self.last_energy = 0.0

    def detect_boundary(self, frame: np.ndarray, is_speech: bool) -> Dict:
        """
        Detect onsets that follow >=200ms silence (turn boundaries)
        """
        frame_duration_ms = len(frame) / self.sr * 1000
        current_energy = np.sqrt(np.mean(frame ** 2))

        if not is_speech:
            self.current_silence_ms += frame_duration_ms
            onset_detected = False
            is_boundary = False
        else:
            # Speech detected
            if self.current_silence_ms >= self.silence_threshold_ms:
                # Onset after long silence = BOUNDARY
                onset_detected = True
                is_boundary = True
            else:
                # Onset but short silence = mid-phrase
                onset_detected = True
                is_boundary = False

            self.current_silence_ms = 0.0

        self.last_energy = current_energy

        return {
            'onset_detected': onset_detected,
            'is_boundary': is_boundary,
            'silence_duration_ms': self.current_silence_ms
        }


class CorticalEntrainment:
    """
    Simulates cortical entrainment to speech rhythm

    Research: Auditory cortex theta oscillations lock to syllable rate (4-8 Hz)
    Once entrained, disruptions in rhythm signal boundaries
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.oscillator = ThetaOscillator(sample_rate=sample_rate)
        self.entrainment_strength = 0.0
        self.predicted_next_peak_time = None

    def extract_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Extract amplitude envelope using Hilbert transform"""
        if len(audio) < 10:
            return np.abs(audio)

        analytic = scipy.signal.hilbert(audio)
        envelope = np.abs(analytic)
        return envelope

    def entrain_to_speech(self, audio: np.ndarray) -> Dict:
        """
        Lock oscillator to speech rhythm
        Returns entrainment strength and predictions
        """
        if len(audio) < 100:
            return {
                'entrainment_strength': 0.0,
                'dominant_freq': 0.0,
                'rhythm_stable': False
            }

        # Extract envelope
        envelope = self.extract_envelope(audio)

        # Find dominant modulation frequency in theta band (4-8 Hz)
        fft = np.abs(np.fft.rfft(envelope))
        freqs = np.fft.rfftfreq(len(envelope), 1/self.sr)

        theta_mask = (freqs >= 4.0) & (freqs <= 8.0)
        if np.sum(theta_mask) == 0:
            return {
                'entrainment_strength': 0.0,
                'dominant_freq': 0.0,
                'rhythm_stable': False
            }

        theta_fft = fft[theta_mask]
        theta_freqs = freqs[theta_mask]

        dominant_freq = theta_freqs[np.argmax(theta_fft)]

        # Adjust oscillator to match
        self.oscillator.set_frequency(dominant_freq)

        # Measure entrainment (peak power in theta band)
        total_power = np.sum(fft ** 2)
        theta_power = np.sum(theta_fft ** 2)

        self.entrainment_strength = theta_power / (total_power + 1e-6)

        # Predict next peak
        period_ms = 1000.0 / dominant_freq
        self.predicted_next_peak_time = time.time() + (period_ms / 1000.0)

        return {
            'entrainment_strength': self.entrainment_strength,
            'dominant_freq': dominant_freq,
            'rhythm_stable': self.entrainment_strength > 0.3
        }

    def detect_rhythm_break(self) -> bool:
        """
        Check if rhythm has been disrupted (boundary signal)
        """
        if self.predicted_next_peak_time is None:
            return False

        # If we're past predicted peak time by >100ms, rhythm broken
        time_error_ms = (time.time() - self.predicted_next_peak_time) * 1000

        return abs(time_error_ms) > 100


class HierarchicalPredictor:
    """
    Multi-timescale hierarchical prediction

    Research: Brain predicts at multiple timescales simultaneously:
    - Acoustic (20ms)
    - Syllable (200ms)
    - Phrase (1s)
    - Turn (3s)
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

        # Multi-timescale energy history
        self.energy_20ms = deque(maxlen=1)     # Acoustic
        self.energy_200ms = deque(maxlen=10)   # Syllable
        self.energy_1s = deque(maxlen=50)      # Phrase
        self.energy_3s = deque(maxlen=150)     # Turn

    def update(self, frame: np.ndarray):
        """Update all timescale buffers"""
        energy = np.sqrt(np.mean(frame ** 2))

        self.energy_20ms.append(energy)
        self.energy_200ms.append(energy)
        self.energy_1s.append(energy)
        self.energy_3s.append(energy)

    def predict_turn_end(self) -> float:
        """
        Hierarchical turn-end prediction

        High-level (turn) predictions constrain low-level predictions
        """
        if len(self.energy_3s) < 50:
            return 0.3  # Not enough context

        # Analyze each timescale
        recent_3s = list(self.energy_3s)
        recent_1s = list(self.energy_1s)
        recent_200ms = list(self.energy_200ms)

        # Turn level: Is energy decreasing over 3s?
        if len(recent_3s) >= 100:
            first_half = np.mean(recent_3s[:len(recent_3s)//2])
            second_half = np.mean(recent_3s[len(recent_3s)//2:])
            turn_decreasing = second_half < first_half * 0.8
        else:
            turn_decreasing = False

        # Phrase level: Is current phrase ending?
        if len(recent_1s) >= 30:
            phrase_energy_drop = recent_1s[-1] < np.mean(recent_1s) * 0.5
        else:
            phrase_energy_drop = False

        # Syllable level: Approaching silence?
        if len(recent_200ms) >= 5:
            syllable_fading = recent_200ms[-1] < np.mean(recent_200ms) * 0.7
        else:
            syllable_fading = False

        # Hierarchical combination
        score = 0.0

        if turn_decreasing:
            score += 0.5  # Turn-level cue (strongest)

        if phrase_energy_drop:
            score += 0.3  # Phrase-level cue

        if syllable_fading:
            score += 0.2  # Syllable-level cue

        return min(score, 1.0)


class NeuromorphicVAD:
    """
    Brain-inspired turn-taking VAD using only audio

    Implements 4 neural mechanisms:
    1. Theta oscillations (temporal segmentation)
    2. STG onset detection (boundary marking)
    3. Cortical entrainment (rhythm tracking)
    4. Hierarchical prediction (multi-timescale context)

    Target: 95-100% accuracy (human-level)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        turn_end_threshold: float = 0.70
    ):
        self.sr = sample_rate
        self.turn_end_threshold = turn_end_threshold

        # Fast speech detection (unchanged)
        self.speech_detector = ProductionVAD(sample_rate=sample_rate)

        # Brain mechanisms (NEW)
        self.theta_oscillator = ThetaOscillator(sample_rate=sample_rate)
        self.onset_detector = STG_OnsetDetector(sample_rate=sample_rate)
        self.entrainment = CorticalEntrainment(sample_rate=sample_rate)
        self.hierarchical = HierarchicalPredictor(sample_rate=sample_rate)

        # Audio buffers
        self.user_buffer = deque(maxlen=sample_rate * 3)  # 3 seconds
        self.ai_buffer = deque(maxlen=sample_rate * 3)

        # Performance tracking
        self.processing_times = deque(maxlen=100)

    def process_frame(
        self,
        user_frame: np.ndarray,
        ai_frame: np.ndarray
    ) -> Dict:
        """
        Process stereo audio frame (user + AI channels)
        """
        start_time = time.perf_counter()

        # Update buffers
        self.user_buffer.extend(user_frame)
        self.ai_buffer.extend(ai_frame)

        # 1. Fast speech detection
        user_result = self.speech_detector.detect_frame(user_frame)
        ai_result = self.speech_detector.detect_frame(ai_frame)

        user_speaking = user_result['is_speech']
        ai_speaking = ai_result['is_speech']

        if not user_speaking:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(latency_ms)
            return {
                'action': 'continue',
                'user_speaking': False,
                'latency_ms': latency_ms
            }

        # 2. Brain mechanisms (only if user speaking)

        # Update hierarchical predictor
        self.hierarchical.update(ai_frame)

        # Theta oscillation prediction
        ai_audio = np.array(list(self.ai_buffer)[-int(self.sr * 0.5):])  # Last 500ms
        if len(ai_audio) > 100:
            envelope = self.entrainment.extract_envelope(ai_audio)

            # Theta phase analysis
            theta_phase = self.theta_oscillator.get_phase()
            # Trough (0 or 2*pi) = boundary, Peak (pi) = mid-phrase
            phase_normalized = abs(theta_phase - np.pi) / np.pi
            theta_turn_end_prob = 1.0 - phase_normalized
        else:
            theta_turn_end_prob = 0.5

        # STG onset detection
        onset_result = self.onset_detector.detect_boundary(ai_frame, ai_speaking)
        is_boundary = onset_result['is_boundary']

        # Cortical entrainment
        if len(self.ai_buffer) >= int(self.sr * 1.0):  # Need 1s for entrainment
            entrainment_result = self.entrainment.entrain_to_speech(
                np.array(list(self.ai_buffer)[-int(self.sr * 1.0):])
            )
            rhythm_stable = entrainment_result['rhythm_stable']
            rhythm_broken = self.entrainment.detect_rhythm_break() if rhythm_stable else False
        else:
            rhythm_stable = False
            rhythm_broken = False

        # Hierarchical prediction
        hierarchical_prob = self.hierarchical.predict_turn_end()

        # FUSION (weighted by neuroscience research)
        final_turn_end_prob = (
            0.35 * theta_turn_end_prob +
            0.25 * (1.0 if is_boundary else 0.0) +
            0.20 * (1.0 if rhythm_broken else 0.0) +
            0.20 * hierarchical_prob
        )

        # Decision
        if ai_speaking and user_speaking:
            # Overlap detected
            if final_turn_end_prob > self.turn_end_threshold:
                # AI finishing turn naturally - DON'T interrupt
                action = "wait_for_ai_completion"
            else:
                # User interrupting mid-sentence - INTERRUPT
                action = "interrupt_ai_immediately"
        elif user_speaking:
            # User speaking, AI silent - clear interruption
            action = "interrupt_ai_immediately"
        else:
            action = "continue"

        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(latency_ms)

        # Advance theta oscillator
        self.theta_oscillator.update(len(user_frame) / self.sr)

        return {
            'action': action,
            'turn_end_prob': final_turn_end_prob,
            'user_speaking': user_speaking,
            'ai_speaking': ai_speaking,
            'overlap': user_speaking and ai_speaking,

            # Neural mechanism contributions
            'theta_contrib': theta_turn_end_prob,
            'onset_boundary': is_boundary,
            'rhythm_stable': rhythm_stable,
            'rhythm_broken': rhythm_broken,
            'hierarchical_contrib': hierarchical_prob,

            # Performance
            'latency_ms': latency_ms
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
        self.user_buffer.clear()
        self.ai_buffer.clear()
        self.onset_detector.current_silence_ms = 0.0
        self.theta_oscillator.phase = 0.0
        self.entrainment.entrainment_strength = 0.0


def demo():
    """Quick demo"""
    print("=" * 80)
    print(" NEUROMORPHIC VAD - Brain-Inspired Turn-Taking Detection")
    print("=" * 80)
    print()
    print("Neural Mechanisms:")
    print("  1. Theta oscillations (4-8 Hz) - 200ms temporal segmentation")
    print("  2. STG onset detection - Boundaries after >=200ms silence")
    print("  3. Cortical entrainment - Rhythm tracking and prediction")
    print("  4. Hierarchical prediction - Multi-timescale context")
    print()
    print("Target: 95-100% accuracy using only audio")
    print()

    sr = 16000
    vad = NeuromorphicVAD(sample_rate=sr)

    # Speed test
    print("Speed Test:")
    print("-" * 80)

    test_frame_user = np.random.randn(160)  # 10ms
    test_frame_ai = np.random.randn(160)

    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        vad.process_frame(test_frame_user, test_frame_ai)
    avg_time = (time.perf_counter() - start) / n_iterations * 1000

    print(f"Average latency: {avg_time:.2f}ms per frame (10ms audio)")
    print(f"Real-time factor: {avg_time / 10:.2f}x")
    print(f"Target: <10ms {'PASS' if avg_time < 10 else 'FAIL'}")
    print()

    stats = vad.get_stats()
    print(f"P50: {stats['p50_latency_ms']:.2f}ms")
    print(f"P95: {stats['p95_latency_ms']:.2f}ms")
    print()
    print("=" * 80)
    print()
    print("Next: Run test_neuromorphic.py for full evaluation on realistic scenarios")
    print()


if __name__ == "__main__":
    demo()
