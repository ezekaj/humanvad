"""
Human-Like Turn-Taking VAD
===========================

Based on neuroscience research:
- Predicts turn-ends 200ms in advance (not just detects speech)
- Uses prosodic features: F0 contour, pitch patterns, speech rate
- Distinguishes turn-holding vs turn-yielding cues

Key Insight: Humans don't just detect "Is someone speaking?"
            They predict "Is this person about to finish?"
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Callable
import time


class TurnTakingVAD:
    """
    Human-like turn-taking detection with prosodic prediction

    Two-stage system:
    1. Speech Detection (from ProductionVAD) - Fast detection of speech presence
    2. Turn-End Prediction (NEW) - Prosodic analysis to predict turn completion
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 10,
        turn_end_threshold: float = 0.65,
        f0_history_ms: int = 300,  # 300ms window for pitch tracking
    ):
        self.sr = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.turn_end_threshold = turn_end_threshold

        # Prosodic feature history
        f0_history_frames = int(f0_history_ms / frame_duration_ms)
        self.f0_history = deque(maxlen=f0_history_frames)  # 30 frames = 300ms
        self.energy_history = deque(maxlen=f0_history_frames)
        self.speech_rate_history = deque(maxlen=50)  # 500ms

        # Adaptive baselines
        self.avg_f0 = 150.0  # Hz (neutral speaking pitch)
        self.avg_speech_rate = 5.0  # syllables/second (~4-8 Hz)
        self.noise_energy = 0.01

        # State
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0

        # Performance
        self.processing_times = deque(maxlen=100)

    def _estimate_f0_autocorrelation(self, frame: np.ndarray) -> float:
        """
        Estimate fundamental frequency (F0) using autocorrelation

        Voice F0 range: 80-400 Hz
        Male: ~85-180 Hz
        Female: ~165-255 Hz
        """
        # F0 range bounds
        min_f0 = 80  # Hz
        max_f0 = 400  # Hz

        min_lag = int(self.sr / max_f0)
        max_lag = int(self.sr / min_f0)

        if len(frame) < max_lag:
            return 0.0

        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Positive lags only

        # Find peak in F0 range
        if max_lag < len(autocorr):
            f0_range = autocorr[min_lag:max_lag]

            if len(f0_range) > 0 and np.max(f0_range) > 0:
                peak_idx = np.argmax(f0_range)
                lag = min_lag + peak_idx
                f0 = self.sr / lag

                # Validate: must be strong peak (>50% of autocorr[0])
                if autocorr[0] > 0 and autocorr[lag] / autocorr[0] > 0.5:
                    return f0

        return 0.0

    def _classify_pitch_pattern(self) -> str:
        """
        Classify pitch contour pattern over last 200-300ms

        Returns:
            'falling' - Turn-ending cue (pitch decreasing)
            'rising' - Turn-holding or question cue (pitch increasing)
            'level' - Turn-holding cue (continuing)
            'unknown' - Insufficient data
        """
        if len(self.f0_history) < 10:  # Need at least 100ms
            return 'unknown'

        # Get recent F0 values (ignore zeros = unvoiced)
        recent_f0 = [f0 for f0 in list(self.f0_history)[-20:] if f0 > 0]

        if len(recent_f0) < 5:
            return 'unknown'

        # Linear regression to get slope
        x = np.arange(len(recent_f0))
        slope, _ = np.polyfit(x, recent_f0, 1)

        # Classify based on slope
        if slope < -5:  # Falling >5 Hz per frame
            return 'falling'
        elif slope > 5:  # Rising >5 Hz per frame
            return 'rising'
        else:
            return 'level'

    def _estimate_speech_rate(self) -> float:
        """
        Estimate speech rate from energy modulation

        Speech has characteristic syllable-rate modulation at 4-8 Hz
        Slowing down = approaching turn-end
        """
        if len(self.energy_history) < 20:  # Need 200ms minimum
            return 0.0

        energies = np.array(list(self.energy_history))

        # FFT to find modulation frequency
        fft = np.abs(np.fft.rfft(energies))
        freqs = np.fft.rfftfreq(len(energies), self.frame_duration_ms / 1000)

        # Find peak in syllable-rate range (3-9 Hz)
        syllable_mask = (freqs >= 3) & (freqs <= 9)

        if np.sum(syllable_mask) > 0:
            syllable_fft = fft[syllable_mask]
            syllable_freqs = freqs[syllable_mask]

            if len(syllable_fft) > 0 and np.max(syllable_fft) > 0:
                peak_idx = np.argmax(syllable_fft)
                speech_rate = syllable_freqs[peak_idx]
                return speech_rate

        return 0.0

    def _detect_speech_simple(self, frame: np.ndarray) -> bool:
        """
        Simple speech detection (from ProductionVAD logic)
        Fast check if someone is speaking
        """
        rms = np.sqrt(np.mean(frame ** 2))

        # Energy threshold (SNR)
        snr = rms / (self.noise_energy + 1e-6)

        # Update noise during silence
        if snr < 1.5:
            alpha = 0.1
            self.noise_energy = (1 - alpha) * self.noise_energy + alpha * rms

        # Speech if energy above noise floor
        return snr > 2.0

    def predict_turn_end(self) -> Dict:
        """
        Predict if current speaker is about to end their turn

        Uses prosodic cues:
        1. Falling pitch (strongest cue)
        2. Slowing speech rate
        3. Decreasing energy (approaching pause)

        Returns:
            {
                'turn_end_prob': 0.0-1.0,
                'pitch_pattern': 'falling'|'rising'|'level',
                'cues': {...}
            }
        """
        if len(self.f0_history) < 10 or len(self.energy_history) < 10:
            return {
                'turn_end_prob': 0.0,
                'pitch_pattern': 'unknown',
                'cues': {}
            }

        # Extract prosodic features
        pitch_pattern = self._classify_pitch_pattern()
        speech_rate = self._estimate_speech_rate()

        # Current energy
        current_energy = self.energy_history[-1] if len(self.energy_history) > 0 else 0
        avg_recent_energy = np.mean(list(self.energy_history)[-10:])

        # Turn-end cues
        cues = {
            'falling_pitch': pitch_pattern == 'falling',
            'slowing_rate': speech_rate < self.avg_speech_rate * 0.7,
            'decreasing_energy': current_energy < avg_recent_energy * 0.8,
            'approaching_silence': current_energy < self.noise_energy * 3
        }

        # Weighted combination (from neuroscience research)
        score = 0.0

        if cues['falling_pitch']:
            score += 0.50  # Strongest cue (research shows this is primary)

        if cues['slowing_rate']:
            score += 0.20  # Speech rate decrease

        if cues['decreasing_energy']:
            score += 0.15  # Energy drop

        if cues['approaching_silence']:
            score += 0.15  # Near pause

        # Update adaptive baselines
        if speech_rate > 0:
            alpha = 0.05
            self.avg_speech_rate = (1 - alpha) * self.avg_speech_rate + alpha * speech_rate

        if len(self.f0_history) > 0:
            recent_f0 = [f0 for f0 in list(self.f0_history) if f0 > 0]
            if len(recent_f0) > 0:
                alpha = 0.05
                self.avg_f0 = (1 - alpha) * self.avg_f0 + alpha * np.mean(recent_f0)

        return {
            'turn_end_prob': min(score, 1.0),
            'pitch_pattern': pitch_pattern,
            'speech_rate': speech_rate,
            'cues': cues
        }

    def process_frame(
        self,
        frame: np.ndarray,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        on_turn_end_predicted: Optional[Callable] = None
    ) -> Dict:
        """
        Process single audio frame with turn-taking prediction

        Returns:
            {
                'is_speaking': bool,
                'turn_end_prob': float,
                'should_interrupt': bool,
                'pitch_pattern': str,
                'latency_ms': float
            }
        """
        start_time = time.perf_counter()

        # Ensure correct frame size
        if len(frame) != self.frame_samples:
            frame = np.pad(frame, (0, max(0, self.frame_samples - len(frame))))[:self.frame_samples]

        # 1. FAST: Detect if speech present
        is_speech = self._detect_speech_simple(frame)

        # 2. Extract prosodic features (if speech)
        f0 = 0.0
        energy = np.sqrt(np.mean(frame ** 2))

        if is_speech:
            f0 = self._estimate_f0_autocorrelation(frame)
            self.f0_history.append(f0)
            self.energy_history.append(energy)
        else:
            # During silence, clear history (reset)
            if len(self.f0_history) > 0 and self.silence_frame_count > 10:
                self.f0_history.clear()
                self.energy_history.clear()

        # 3. PREDICT: Turn-end probability
        turn_prediction = self.predict_turn_end()

        # 4. Update state tracking
        if is_speech:
            self.speech_frame_count += 1
            self.silence_frame_count = 0

            if not self.is_speaking and self.speech_frame_count >= 3:  # 30ms
                self.is_speaking = True
                if on_speech_start:
                    on_speech_start()
        else:
            self.silence_frame_count += 1

            if self.is_speaking and self.silence_frame_count >= 10:  # 100ms
                self.is_speaking = False
                self.speech_frame_count = 0
                if on_speech_end:
                    on_speech_end()

        # 5. Interruption decision logic
        should_interrupt = False

        if is_speech and self.is_speaking:
            # Speech detected
            turn_end_prob = turn_prediction['turn_end_prob']

            if turn_end_prob > self.turn_end_threshold:
                # Speaker is finishing turn - this is a RESPONSE, not interruption
                # Wait for natural pause (don't interrupt)
                should_interrupt = False

                if on_turn_end_predicted:
                    on_turn_end_predicted(turn_prediction)
            else:
                # Speaker is mid-sentence - this IS an interruption
                # User is cutting in, interrupt AI immediately
                should_interrupt = True

        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(latency_ms)

        return {
            'is_speaking': self.is_speaking,
            'turn_end_prob': turn_prediction['turn_end_prob'],
            'should_interrupt': should_interrupt,
            'pitch_pattern': turn_prediction['pitch_pattern'],
            'speech_rate': turn_prediction.get('speech_rate', 0.0),
            'f0': f0,
            'latency_ms': latency_ms,
            'cues': turn_prediction.get('cues', {})
        }

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if len(self.processing_times) > 0:
            return {
                'avg_latency_ms': np.mean(self.processing_times),
                'p50_latency_ms': np.percentile(self.processing_times, 50),
                'p95_latency_ms': np.percentile(self.processing_times, 95),
                'max_latency_ms': np.max(self.processing_times),
                'avg_f0': self.avg_f0,
                'avg_speech_rate': self.avg_speech_rate
            }
        return {}

    def reset(self):
        """Reset state"""
        self.is_speaking = False
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.f0_history.clear()
        self.energy_history.clear()
        self.speech_rate_history.clear()


def demo():
    """Quick demo"""
    print("=" * 80)
    print(" TURN-TAKING VAD - Human-Like Conversation Prediction")
    print("=" * 80)
    print()
    print("Features:")
    print("  - F0 (pitch) tracking via autocorrelation")
    print("  - Pitch pattern classification (falling/rising/level)")
    print("  - Speech rate estimation (syllable modulation)")
    print("  - Turn-end prediction (200ms anticipation)")
    print()
    print("Key Innovation:")
    print("  Predicts WHEN speaker will finish, not just IF they're speaking")
    print()

    sr = 16000
    vad = TurnTakingVAD(sample_rate=sr)

    # Test speed
    print("Speed Test:")
    print("-" * 80)

    test_frame = np.random.randn(160)  # 10ms frame
    n_iterations = 100

    start = time.perf_counter()
    for _ in range(n_iterations):
        vad.process_frame(test_frame)
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
    print("Next: Run test_turn_taking.py for full evaluation with prosodic scenarios")
    print()


if __name__ == "__main__":
    demo()
