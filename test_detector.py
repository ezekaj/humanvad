"""
Comprehensive test suite for speech detector
"""

import numpy as np
from speech_detector import HumanInspiredSpeechDetector


def generate_speech_like(duration=0.5, sr=16000):
    """Generate realistic speech-like signal"""
    t = np.linspace(0, duration, int(sr * duration))

    # Variable F0 (pitch contour - typical speech)
    f0_start = 120  # Start pitch
    f0_end = 180    # End pitch (rising intonation)
    f0 = np.linspace(f0_start, f0_end, len(t))

    # Fundamental + strong harmonics (more realistic for voiced speech)
    signal = 1.0 * np.sin(2 * np.pi * f0 * t)  # Fundamental
    signal += 0.8 * np.sin(2 * np.pi * 2 * f0 * t)  # 2nd harmonic (strong)
    signal += 0.6 * np.sin(2 * np.pi * 3 * f0 * t)  # 3rd harmonic (strong)
    signal += 0.4 * np.sin(2 * np.pi * 4 * f0 * t)  # 4th harmonic
    signal += 0.3 * np.sin(2 * np.pi * 5 * f0 * t)  # 5th harmonic

    # Formant-like resonances
    # F1 (500 Hz)
    formant1 = 0.4 * np.sin(2 * np.pi * 500 * t)
    # F2 (1500 Hz)
    formant2 = 0.3 * np.sin(2 * np.pi * 1500 * t)
    # F3 (2500 Hz)
    formant3 = 0.2 * np.sin(2 * np.pi * 2500 * t)

    signal += formant1 + formant2 + formant3

    # Syllable-rate modulation (4 Hz - 4 syllables/sec)
    syllable_rate = 4
    modulation = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t))
    signal *= modulation

    # Add pauses (word boundaries)
    pause_duration = int(0.1 * sr)  # 100ms pauses
    for pause_start in [int(0.33 * len(t)), int(0.66 * len(t))]:
        signal[pause_start:pause_start + pause_duration] *= 0.1

    # Normalize
    signal = signal / np.max(np.abs(signal))

    return signal


def generate_white_noise(duration=0.5, sr=16000):
    """Generate white noise"""
    return np.random.randn(int(sr * duration))


def generate_pink_noise(duration=0.5, sr=16000):
    """Generate pink noise (1/f spectrum)"""
    n_samples = int(sr * duration)
    noise = np.random.randn(n_samples)

    # Apply 1/f filter
    from scipy import signal as sp_signal
    b, a = sp_signal.butter(1, 0.1, btype='low')
    pink = sp_signal.filtfilt(b, a, noise)

    return pink / np.max(np.abs(pink))


def generate_pure_tone(freq=1000, duration=0.5, sr=16000):
    """Generate pure sine tone"""
    t = np.linspace(0, duration, int(sr * duration))
    return np.sin(2 * np.pi * freq * t)


def generate_music_like(duration=0.5, sr=16000):
    """Generate music-like signal (harmonic but not speech)"""
    t = np.linspace(0, duration, int(sr * duration))

    # Musical notes (C major chord: C4, E4, G4)
    c4 = 261.63
    e4 = 329.63
    g4 = 392.00

    signal = (
        np.sin(2 * np.pi * c4 * t) +
        np.sin(2 * np.pi * e4 * t) +
        np.sin(2 * np.pi * g4 * t)
    )

    # Add harmonics for each note
    for note in [c4, e4, g4]:
        signal += 0.5 * np.sin(2 * np.pi * 2 * note * t)
        signal += 0.3 * np.sin(2 * np.pi * 3 * note * t)

    # No syllable-rate modulation (key difference from speech)
    # Instead, slow vibrato (6 Hz)
    vibrato = 0.95 + 0.05 * np.sin(2 * np.pi * 6 * t)
    signal *= vibrato

    return signal / np.max(np.abs(signal))


def generate_click_noise(duration=0.5, sr=16000):
    """Generate clicking/typing noise"""
    n_samples = int(sr * duration)
    signal = np.zeros(n_samples)

    # Random clicks
    n_clicks = int(duration * 10)  # 10 clicks per second
    click_positions = np.random.randint(0, n_samples, n_clicks)

    for pos in click_positions:
        # Short impulse
        if pos < n_samples - 100:
            signal[pos:pos + 100] = np.random.randn(100) * 0.5

    return signal


def generate_hum_noise(duration=0.5, sr=16000):
    """Generate electrical hum (60 Hz + harmonics)"""
    t = np.linspace(0, duration, int(sr * duration))

    signal = (
        np.sin(2 * np.pi * 60 * t) +
        0.5 * np.sin(2 * np.pi * 120 * t) +
        0.3 * np.sin(2 * np.pi * 180 * t)
    )

    return signal


def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("=" * 80)
    print(" COMPREHENSIVE SPEECH DETECTION TEST SUITE")
    print("=" * 80)
    print()

    detector = HumanInspiredSpeechDetector()

    # Test cases: (name, signal, expected_result)
    test_cases = [
        ("Speech-like (realistic)", generate_speech_like(), True),
        ("White noise", generate_white_noise(), False),
        ("Pink noise (1/f)", generate_pink_noise(), False),
        ("Pure tone (1 kHz)", generate_pure_tone(1000), False),
        ("Pure tone (150 Hz - speech F0)", generate_pure_tone(150), False),
        ("Music-like (harmonic)", generate_music_like(), False),
        ("Click/typing noise", generate_click_noise(), False),
        ("Electrical hum (60 Hz)", generate_hum_noise(), False),
    ]

    results = []
    print(f"{'Test Case':<35} {'Expected':<10} {'Detected':<10} {'Confidence':<12} {'Result'}")
    print("-" * 80)

    for name, signal, expected in test_cases:
        result = detector.detect(signal)

        detected = result['is_speech']
        confidence = result['confidence']

        # Check if correct
        correct = (detected == expected)
        status = "PASS" if correct else "FAIL"

        print(f"{name:<35} {str(expected):<10} {str(detected):<10} {confidence:>6.1%}      {status}")

        results.append({
            'name': name,
            'expected': expected,
            'detected': detected,
            'confidence': confidence,
            'correct': correct
        })

    print()
    print("=" * 80)

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100

    print(f" SUMMARY: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("=" * 80)
    print()

    # Detailed analysis of failures
    failures = [r for r in results if not r['correct']]
    if failures:
        print("FAILURES:")
        print("-" * 80)
        for fail in failures:
            print(f"  {fail['name']}")
            print(f"    Expected: {fail['expected']}, Got: {fail['detected']}")
            print(f"    Confidence: {fail['confidence']:.1%}")
        print()

    # Feature analysis for interesting cases
    print("DETAILED FEATURE ANALYSIS:")
    print("-" * 80)

    interesting_cases = [
        ("Speech-like (realistic)", generate_speech_like()),
        ("Music-like (harmonic)", generate_music_like()),
        ("White noise", generate_white_noise()),
    ]

    for name, signal in interesting_cases:
        result = detector.detect(signal)
        features = result['features']

        print(f"\n{name}:")
        print(f"  Decision: {'SPEECH' if result['is_speech'] else 'NOT SPEECH'} "
              f"(confidence: {result['confidence']:.1%}, score: {result['score']:.2f})")
        print(f"  F0: {features['f0']:.1f} Hz")
        print(f"  Harmonicity: {features['harmonicity']:.2f}")
        print(f"  Has formants: {features['has_formants']} (n={features['n_formants']})")
        print(f"  Modulation ratio: {features['modulation_ratio']:.3f}")
        print(f"  Phase coherence: {features['phase_coherence']:.2f}")
        print(f"  Onset strength: {features['onset_strength']:.4f}")

    print()
    print("=" * 80)

    return accuracy


if __name__ == "__main__":
    accuracy = run_comprehensive_tests()

    if accuracy >= 85:
        print(f"TEST PASSED: {accuracy:.1f}% accuracy (target: >=85%)")
    else:
        print(f"TEST FAILED: {accuracy:.1f}% accuracy (target: >=85%)")
