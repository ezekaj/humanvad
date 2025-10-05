"""
Human-Inspired Speech Detection System
Based on neuroscience research of human auditory processing

Implements multi-layer detection mimicking human brain:
1. Cochlear filtering (frequency decomposition)
2. Brainstem processing (pitch, onsets)
3. Auditory cortex (formants, modulation)
4. Superior temporal gyrus (phase locking)
5. Attention mechanism (context prediction)
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SpeechDetectionConfig:
    """Configuration for speech detection"""
    sample_rate: int = 16000

    # Frequency ranges
    f0_min: float = 85.0    # Min fundamental frequency (Hz)
    f0_max: float = 255.0   # Max fundamental frequency (Hz)

    # Formant ranges (Hz)
    f1_range: Tuple[float, float] = (300, 800)
    f2_range: Tuple[float, float] = (800, 2500)
    f3_range: Tuple[float, float] = (2000, 3500)

    # Temporal modulation (syllable rate in Hz)
    syllable_rate_min: float = 3.0
    syllable_rate_max: float = 8.0

    # Detection thresholds
    harmonicity_threshold: float = 0.25  # Lower for varying F0 (realistic speech)
    phase_coherence_threshold: float = 0.7
    modulation_threshold: float = 0.6  # Stricter threshold for syllable-rate energy

    # Processing
    n_mels: int = 40
    frame_length: int = 512
    hop_length: int = 256


class HumanInspiredSpeechDetector:
    """
    Speech detector mimicking human auditory processing
    """

    def __init__(self, config: Optional[SpeechDetectionConfig] = None):
        self.config = config or SpeechDetectionConfig()
        self.sr = self.config.sample_rate

        # Build mel filterbank (mimics cochlea)
        self.mel_filters = self._create_mel_filterbank()

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel-scale filterbank (mimics cochlear filtering)"""
        n_fft = self.config.frame_length
        n_mels = self.config.n_mels

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel points
        min_mel = hz_to_mel(0)
        max_mel = hz_to_mel(self.sr / 2)
        mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bins
        bin_points = np.floor((n_fft + 1) * hz_points / self.sr).astype(int)

        # Create filterbank
        filters = np.zeros((n_mels, n_fft // 2 + 1))

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Triangular filter
            for j in range(left, center):
                filters[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filters[i, j] = (right - j) / (right - center)

        return filters

    def layer1_cochlear_filtering(self, audio: np.ndarray) -> np.ndarray:
        """
        Layer 1: Cochlear filtering (0-20ms)
        Mimics frequency decomposition in the cochlea

        Returns: Mel-scale spectrogram
        """
        # Compute STFT
        f, t, Zxx = signal.stft(
            audio,
            fs=self.sr,
            nperseg=self.config.frame_length,
            noverlap=self.config.frame_length - self.config.hop_length
        )

        # Power spectrogram
        power_spec = np.abs(Zxx) ** 2

        # Apply mel filterbank
        mel_spec = self.mel_filters @ power_spec

        # Compression (mimics outer hair cells)
        compressed = mel_spec ** 0.3

        return compressed

    def layer2_brainstem_processing(self, audio: np.ndarray) -> Dict:
        """
        Layer 2: Brainstem processing (20-50ms)
        Extracts pitch and onset information

        Returns: Dict with f0, harmonicity, onsets
        """
        # Autocorrelation for pitch detection
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags

        # Normalize
        autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr

        # Find peaks (harmonic structure)
        min_lag = int(self.sr / self.config.f0_max)
        max_lag = int(self.sr / self.config.f0_min)

        search_region = autocorr[min_lag:max_lag]

        if len(search_region) > 0:
            peak_idx = np.argmax(search_region)
            f0 = self.sr / (min_lag + peak_idx)

            # Harmonicity (how regular are the peaks)
            peak_value = search_region[peak_idx]
            harmonicity = peak_value
        else:
            f0 = 0
            harmonicity = 0

        # Onset detection (energy changes)
        envelope = np.abs(signal.hilbert(audio))
        onset_strength = np.diff(envelope)
        onset_strength = np.maximum(onset_strength, 0)

        return {
            'f0': f0,
            'harmonicity': harmonicity,
            'onset_strength': np.mean(onset_strength),
            'has_speech_pitch': self.config.f0_min <= f0 <= self.config.f0_max
        }

    def layer3_cortical_processing(self, mel_spec: np.ndarray) -> Dict:
        """
        Layer 3: Auditory cortex processing (50-150ms)
        Detects formants and temporal modulation

        Returns: Dict with formants, modulation energy
        """
        # Average spectrum (for formant detection)
        avg_spectrum = np.mean(mel_spec, axis=1)

        # Find spectral peaks (formants)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(avg_spectrum, height=np.mean(avg_spectrum))

        # Check if formant structure exists
        has_formants = len(peaks) >= 2

        # Temporal modulation analysis
        # Compute modulation spectrum (FFT of temporal envelope)
        temporal_envelopes = mel_spec  # Each row is temporal envelope of one frequency band

        modulation_energies = []
        for envelope in temporal_envelopes:
            # FFT of envelope
            mod_spectrum = np.abs(fft(envelope))
            mod_freqs = fftfreq(len(envelope), d=1.0 / (self.sr / self.config.hop_length))

            # Energy in syllable rate (3-8 Hz)
            syllable_mask = (mod_freqs >= self.config.syllable_rate_min) & \
                          (mod_freqs <= self.config.syllable_rate_max)

            if np.any(syllable_mask):
                syllable_energy = np.mean(mod_spectrum[syllable_mask])
            else:
                syllable_energy = 0

            modulation_energies.append(syllable_energy)

        avg_modulation_energy = np.mean(modulation_energies)
        total_energy = np.mean(mel_spec) + 1e-6

        modulation_ratio = avg_modulation_energy / total_energy

        return {
            'has_formants': has_formants,
            'n_formants': len(peaks),
            'modulation_ratio': modulation_ratio,
            'syllable_energy': avg_modulation_energy
        }

    def layer4_phase_locking(self, audio: np.ndarray) -> Dict:
        """
        Layer 4: Superior Temporal Gyrus (150-300ms)
        Phase locking to speech rhythm

        Returns: Dict with phase coherence
        """
        # Extract envelope
        analytic_signal = signal.hilbert(audio)
        envelope = np.abs(analytic_signal)

        # Bandpass filter envelope at syllable rate (3-8 Hz)
        sos = signal.butter(
            4,
            [self.config.syllable_rate_min, self.config.syllable_rate_max],
            btype='band',
            fs=self.sr,
            output='sos'
        )

        syllable_envelope = signal.sosfilt(sos, envelope)

        # Compute "phase coherence" (simplified: autocorrelation of filtered envelope)
        env_autocorr = np.correlate(syllable_envelope, syllable_envelope, mode='same')
        env_autocorr = env_autocorr / (env_autocorr[len(env_autocorr) // 2] + 1e-6)

        # Peak in autocorrelation = rhythmic structure
        center = len(env_autocorr) // 2
        search_range = int(self.sr * 0.3)  # Search ±300ms

        search_region = env_autocorr[center - search_range:center + search_range]
        if len(search_region) > 0:
            max_coherence = np.max(np.abs(search_region))
        else:
            max_coherence = 0

        return {
            'phase_coherence': max_coherence,
            'has_rhythm': max_coherence > self.config.phase_coherence_threshold
        }

    def layer5_attention_mechanism(self, features: Dict, context: Optional[Dict] = None) -> float:
        """
        Layer 5: Frontal cortex attention (200-500ms)
        Top-down prediction and attention gating

        Returns: Attention boost factor
        """
        # Simplified: Use previous detection as context
        if context and context.get('was_speech', False):
            # More likely to be speech if previous frame was speech
            attention_boost = 1.2
        else:
            attention_boost = 1.0

        # Additional boost if features are very speech-like
        if features.get('has_speech_pitch') and features.get('has_formants'):
            attention_boost *= 1.1

        return attention_boost

    def detect(self, audio: np.ndarray, context: Optional[Dict] = None) -> Dict:
        """
        Full multi-layer speech detection pipeline

        Args:
            audio: Audio signal (numpy array)
            context: Optional context from previous detection

        Returns:
            Dict with detection result and all features
        """
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.flatten()

        # Layer 1: Cochlear filtering
        mel_spec = self.layer1_cochlear_filtering(audio)

        # Layer 2: Brainstem processing
        brainstem_features = self.layer2_brainstem_processing(audio)

        # Layer 3: Cortical processing
        cortical_features = self.layer3_cortical_processing(mel_spec)

        # Layer 4: Phase locking
        phase_features = self.layer4_phase_locking(audio)

        # Combine all features
        all_features = {
            **brainstem_features,
            **cortical_features,
            **phase_features
        }

        # Layer 5: Attention mechanism
        attention = self.layer5_attention_mechanism(all_features, context)

        # Decision fusion (multi-cue integration)
        score = 0.0
        confidence_breakdown = {}

        # Cue 1: Pitch in speech range (weight: 2.5) - REQUIRED
        if brainstem_features['has_speech_pitch']:
            pitch_score = 2.5
            score += pitch_score
            confidence_breakdown['pitch'] = pitch_score
        else:
            confidence_breakdown['pitch'] = 0
            # No speech pitch = automatic rejection
            score -= 5.0

        # Cue 2: Harmonicity (weight: 2.0) - Must be present AND strong
        if brainstem_features['harmonicity'] > self.config.harmonicity_threshold:
            # Scale by harmonicity strength
            harmony_score = 2.0 * brainstem_features['harmonicity']
            score += harmony_score
            confidence_breakdown['harmonicity'] = harmony_score
        else:
            confidence_breakdown['harmonicity'] = 0
            # Weak harmonicity = penalty
            score -= 1.0

        # Cue 3: Formants present (weight: 2.0) - REQUIRED for speech
        if cortical_features['has_formants'] and cortical_features['n_formants'] >= 2:
            formant_score = 2.0
            score += formant_score
            confidence_breakdown['formants'] = formant_score
        else:
            confidence_breakdown['formants'] = 0
            # No formants = major penalty
            score -= 3.0

        # Cue 4: Syllable-rate modulation (weight: 2.5) - Critical for speech
        mod_ratio = cortical_features['modulation_ratio']
        if mod_ratio > self.config.modulation_threshold:
            # Cap modulation contribution (noise can have high modulation)
            mod_score = 2.5 * min(mod_ratio / 1.0, 1.0)  # Cap at ratio=1.0
            score += mod_score
            confidence_breakdown['modulation'] = mod_score
        else:
            confidence_breakdown['modulation'] = 0
            score -= 1.0

        # Cue 5: Phase coherence (weight: 1.5) - Useful but can be noisy
        if phase_features['has_rhythm']:
            # Reduce weight to avoid over-reliance
            rhythm_score = 1.5 * min(phase_features['phase_coherence'], 1.0)
            score += rhythm_score
            confidence_breakdown['rhythm'] = rhythm_score
        else:
            confidence_breakdown['rhythm'] = 0

        # Apply attention
        score *= attention

        # Normalize to 0-1 (allow negative scores)
        max_score = 10.5  # Maximum possible score
        min_score = -10.0  # Minimum possible score
        confidence = max(0, min((score - min_score) / (max_score - min_score), 1.0))

        # Stricter decision threshold - ALL key features must be present
        is_speech = (
            score >= 7.0 and  # Higher threshold
            brainstem_features['has_speech_pitch'] and  # MUST have speech pitch
            cortical_features['has_formants'] and  # MUST have formants
            brainstem_features['harmonicity'] > self.config.harmonicity_threshold and  # MUST be harmonic
            cortical_features['modulation_ratio'] > 3.0  # MUST have strong syllable-rate modulation (key for speech vs music)
        )

        return {
            'is_speech': is_speech,
            'confidence': confidence,
            'score': score,
            'confidence_breakdown': confidence_breakdown,
            'features': all_features,
            'attention_boost': attention
        }

    def detect_streaming(self, audio_stream: np.ndarray, frame_size: int = 16000) -> list:
        """
        Detect speech in streaming audio (frame-by-frame)

        Args:
            audio_stream: Long audio signal
            frame_size: Size of each frame (default: 1 second at 16kHz)

        Returns:
            List of detection results per frame
        """
        results = []
        context = None

        n_frames = len(audio_stream) // frame_size

        for i in range(n_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_stream[start:end]

            # Detect
            result = self.detect(frame, context)

            # Update context for next frame
            context = {
                'was_speech': result['is_speech'],
                'confidence': result['confidence']
            }

            results.append({
                'frame_idx': i,
                'time_start': start / self.sr,
                'time_end': end / self.sr,
                **result
            })

        return results


def demo():
    """Demo: Create synthetic speech-like and noise signals"""
    print("=" * 70)
    print(" Human-Inspired Speech Detection Demo")
    print("=" * 70)
    print()

    detector = HumanInspiredSpeechDetector()

    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Test 1: Synthetic speech-like signal
    print("Test 1: Synthetic Speech-like Signal")
    print("-" * 70)

    # Fundamental frequency (150 Hz - typical male voice)
    f0 = 150
    speech = np.sin(2 * np.pi * f0 * t)  # F0

    # Add harmonics
    speech += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)  # 2nd harmonic
    speech += 0.3 * np.sin(2 * np.pi * 3 * f0 * t)  # 3rd harmonic

    # Add syllable-rate modulation (5 Hz)
    syllable_rate = 5
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t)
    speech *= modulation

    # Detect
    result = detector.detect(speech)

    print(f"Is Speech: {result['is_speech']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Score: {result['score']:.2f}/10")
    print(f"\nFeature Breakdown:")
    print(f"  F0: {result['features']['f0']:.1f} Hz (target: {f0} Hz)")
    print(f"  Harmonicity: {result['features']['harmonicity']:.2f}")
    print(f"  Has formants: {result['features']['has_formants']}")
    print(f"  Modulation ratio: {result['features']['modulation_ratio']:.3f}")
    print(f"  Phase coherence: {result['features']['phase_coherence']:.2f}")
    print()

    # Test 2: White noise
    print("Test 2: White Noise")
    print("-" * 70)

    noise = np.random.randn(len(t))

    result = detector.detect(noise)

    print(f"Is Speech: {result['is_speech']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Score: {result['score']:.2f}/10")
    print(f"\nFeature Breakdown:")
    print(f"  F0: {result['features']['f0']:.1f} Hz")
    print(f"  Harmonicity: {result['features']['harmonicity']:.2f}")
    print(f"  Has formants: {result['features']['has_formants']}")
    print(f"  Modulation ratio: {result['features']['modulation_ratio']:.3f}")
    print(f"  Phase coherence: {result['features']['phase_coherence']:.2f}")
    print()

    # Test 3: Pure tone (not speech)
    print("Test 3: Pure Tone (1 kHz)")
    print("-" * 70)

    tone = np.sin(2 * np.pi * 1000 * t)

    result = detector.detect(tone)

    print(f"Is Speech: {result['is_speech']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Score: {result['score']:.2f}/10")
    print(f"\nFeature Breakdown:")
    print(f"  F0: {result['features']['f0']:.1f} Hz")
    print(f"  Harmonicity: {result['features']['harmonicity']:.2f}")
    print(f"  Has formants: {result['features']['has_formants']}")
    print(f"  Modulation ratio: {result['features']['modulation_ratio']:.3f}")
    print(f"  Phase coherence: {result['features']['phase_coherence']:.2f}")
    print()

    print("=" * 70)
    print(" Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    demo()
