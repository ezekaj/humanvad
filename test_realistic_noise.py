"""
Test VAD with realistic background noise scenarios
"""

import numpy as np
from silero_vad_wrapper import SileroVAD


def generate_speech_like(duration=0.5, sr=16000):
    """Generate realistic speech-like signal"""
    t = np.linspace(0, duration, int(sr * duration))

    # Variable F0 (pitch contour)
    f0_start = 120
    f0_end = 180
    f0 = np.linspace(f0_start, f0_end, len(t))

    # Fundamental + harmonics
    signal = 1.0 * np.sin(2 * np.pi * f0 * t)
    signal += 0.8 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.6 * np.sin(2 * np.pi * 3 * f0 * t)

    # Formants
    signal += 0.4 * np.sin(2 * np.pi * 500 * t)
    signal += 0.3 * np.sin(2 * np.pi * 1500 * t)

    # Syllable-rate modulation
    modulation = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    signal *= modulation

    return signal / np.max(np.abs(signal))


def add_background_noise(speech, noise, snr_db):
    """
    Add background noise to speech at specific SNR

    SNR (Signal-to-Noise Ratio):
    - 20 dB = very quiet background (library)
    - 10 dB = normal conversation (office)
    - 5 dB = noisy environment (restaurant)
    - 0 dB = very noisy (loud bar, street)
    - -5 dB = extreme noise (construction, club)
    """
    # Calculate RMS of speech and noise
    speech_rms = np.sqrt(np.mean(speech ** 2))
    noise_rms = np.sqrt(np.mean(noise ** 2))

    # Calculate desired noise level for target SNR
    # SNR = 20 * log10(signal_rms / noise_rms)
    # noise_rms_target = signal_rms / (10 ^ (SNR/20))
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))

    # Scale noise to target level
    noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-10))

    # Mix
    mixed = speech + noise_scaled

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed


def generate_white_noise(duration=0.5, sr=16000):
    """White noise (random)"""
    return np.random.randn(int(sr * duration)) * 0.1


def generate_babble_noise(duration=0.5, sr=16000):
    """Multiple people talking (cocktail party)"""
    n_samples = int(sr * duration)
    babble = np.zeros(n_samples)

    # Simulate 5-10 background speakers
    n_speakers = np.random.randint(5, 10)

    for _ in range(n_speakers):
        # Random pitch for each speaker
        f0 = 80 + np.random.rand() * 150
        t = np.linspace(0, duration, n_samples)

        # Basic harmonic structure
        speaker = np.sin(2 * np.pi * f0 * t)
        speaker += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)

        # Random amplitude modulation
        mod_rate = 2 + np.random.rand() * 4
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_rate * t)
        speaker *= modulation

        # Random amplitude for each speaker
        speaker *= 0.1 + np.random.rand() * 0.3

        babble += speaker

    return babble / (np.max(np.abs(babble)) + 1e-10)


def generate_music_noise(duration=0.5, sr=16000):
    """Background music"""
    t = np.linspace(0, duration, int(sr * duration))

    # Chord progression (4 notes)
    notes = [261.63, 329.63, 392.00, 440.00]  # C, E, G, A

    music = np.zeros(len(t))
    for note in notes:
        music += np.sin(2 * np.pi * note * t)
        music += 0.5 * np.sin(2 * np.pi * 2 * note * t)

    # Add rhythm (4 Hz beat)
    rhythm = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 4 * t))
    music *= rhythm

    return music / np.max(np.abs(music))


def generate_traffic_noise(duration=0.5, sr=16000):
    """Street traffic noise"""
    n_samples = int(sr * duration)

    # Low-frequency rumble (engines)
    t = np.linspace(0, duration, n_samples)
    rumble = (
        np.sin(2 * np.pi * 40 * t) +
        0.5 * np.sin(2 * np.pi * 60 * t) +
        0.3 * np.sin(2 * np.pi * 80 * t)
    )

    # High-frequency noise (tires, wind)
    hiss = np.random.randn(n_samples) * 0.3

    # Random horn honks
    for _ in range(2):
        start = np.random.randint(0, n_samples - 1000)
        honk = np.sin(2 * np.pi * 400 * np.arange(1000) / sr)
        rumble[start:start+1000] += honk * 0.5

    traffic = rumble + hiss
    return traffic / (np.max(np.abs(traffic)) + 1e-10)


def generate_tv_noise(duration=0.5, sr=16000):
    """TV/radio in background"""
    # Simulate speech-like content (but not real speech)
    t = np.linspace(0, duration, int(sr * duration))

    # Varying pitch (TV announcer)
    f0 = 100 + 30 * np.sin(2 * np.pi * 0.5 * t)

    tv = np.sin(2 * np.pi * f0 * t)
    tv += 0.6 * np.sin(2 * np.pi * 2 * f0 * t)

    # Modulation
    mod = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 3 * t))
    tv *= mod

    # Add static/compression artifacts
    tv += np.random.randn(len(t)) * 0.1

    return tv / np.max(np.abs(tv))


def run_realistic_tests():
    """Test VAD with realistic background noise"""
    print("=" * 80)
    print(" REALISTIC NOISE TEST - Person Speaking vs Background Noise")
    print("=" * 80)
    print()

    sr = 16000
    vad = SileroVAD(sample_rate=sr)

    # Generate clean speech
    speech = generate_speech_like(duration=0.5, sr=sr)

    # Test scenarios
    print("Testing speech detection in various noise conditions:")
    print("-" * 80)
    print(f"{'Scenario':<40} {'SNR':<10} {'Detected':<12} {'Confidence':<12} {'Result'}")
    print("-" * 80)

    test_cases = []

    # 1. Clean speech (baseline)
    test_cases.append(("Clean speech (no noise)", speech, None, True))

    # 2. Speech + White noise at various SNRs
    for snr in [20, 10, 5, 0, -5]:
        noise = generate_white_noise(duration=0.5, sr=sr)
        mixed = add_background_noise(speech, noise, snr)
        test_cases.append((f"Speech + white noise", mixed, f"{snr} dB", True))

    # 3. Speech + Babble (cocktail party)
    for snr in [10, 5, 0, -5]:
        babble = generate_babble_noise(duration=0.5, sr=sr)
        mixed = add_background_noise(speech, babble, snr)
        test_cases.append((f"Speech + babble (party)", mixed, f"{snr} dB", True))

    # 4. Speech + Music
    for snr in [10, 5, 0]:
        music = generate_music_noise(duration=0.5, sr=sr)
        mixed = add_background_noise(speech, music, snr)
        test_cases.append((f"Speech + music", mixed, f"{snr} dB", True))

    # 5. Speech + Traffic
    for snr in [10, 5, 0]:
        traffic = generate_traffic_noise(duration=0.5, sr=sr)
        mixed = add_background_noise(speech, traffic, snr)
        test_cases.append((f"Speech + traffic", mixed, f"{snr} dB", True))

    # 6. Speech + TV
    for snr in [10, 5, 0]:
        tv = generate_tv_noise(duration=0.5, sr=sr)
        mixed = add_background_noise(speech, tv, snr)
        test_cases.append((f"Speech + TV background", mixed, f"{snr} dB", True))

    # 7. JUST NOISE (should be rejected)
    test_cases.append(("White noise ONLY (no speech)", generate_white_noise(0.5, sr), "N/A", False))
    test_cases.append(("Babble ONLY (no speech)", generate_babble_noise(0.5, sr), "N/A", False))
    test_cases.append(("Music ONLY (no speech)", generate_music_noise(0.5, sr), "N/A", False))
    test_cases.append(("Traffic ONLY (no speech)", generate_traffic_noise(0.5, sr), "N/A", False))
    test_cases.append(("TV ONLY (no speech)", generate_tv_noise(0.5, sr), "N/A", False))

    # Run tests
    results = []
    for name, signal, snr_label, expected in test_cases:
        # Silero VAD requires full signal (or chunks >=512 samples)
        # Test on entire 0.5s signal at once
        result = vad.detect_frame(signal)
        avg_confidence = result['confidence']
        detected = result['is_speech']

        # Check correctness
        correct = (detected == expected)
        status = "PASS" if correct else "FAIL"

        snr_str = snr_label if snr_label else "N/A"

        print(f"{name:<40} {snr_str:<10} {str(detected):<12} {avg_confidence:>6.1%}      {status}")

        results.append({
            'name': name,
            'expected': expected,
            'detected': detected,
            'confidence': avg_confidence,
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

    # Detailed failures
    failures = [r for r in results if not r['correct']]
    if failures:
        print("FAILURES:")
        print("-" * 80)
        for fail in failures:
            print(f"  {fail['name']}")
            print(f"    Expected: {fail['expected']}, Got: {fail['detected']}")
            print(f"    Confidence: {fail['confidence']:.1%}")
        print()

    # SNR analysis (speech detection at different noise levels)
    print("SNR PERFORMANCE ANALYSIS:")
    print("-" * 80)

    snr_groups = {}
    for r in results:
        if 'dB' in r['name']:
            # Extract SNR
            parts = r['name'].split()
            for i, part in enumerate(parts):
                if 'dB' in part:
                    snr = parts[i-1]
                    if snr not in snr_groups:
                        snr_groups[snr] = {'total': 0, 'correct': 0}
                    snr_groups[snr]['total'] += 1
                    if r['correct']:
                        snr_groups[snr]['correct'] += 1

    for snr in sorted(snr_groups.keys(), key=lambda x: float(x), reverse=True):
        stats = snr_groups[snr]
        acc = stats['correct'] / stats['total'] * 100
        print(f"  SNR {snr:>4} dB: {stats['correct']}/{stats['total']} correct ({acc:.1f}%)")

    print()
    print("=" * 80)
    print()

    if accuracy >= 85:
        print(f"TEST PASSED: {accuracy:.1f}% accuracy (target: >=85%)")
    else:
        print(f"TEST FAILED: {accuracy:.1f}% accuracy (target: >=85%)")

    print()
    print("KEY INSIGHTS:")
    print("  - SNR > 10 dB: Should detect speech reliably (quiet background)")
    print("  - SNR 5-10 dB: Moderate noise (normal conversation)")
    print("  - SNR 0-5 dB: High noise (challenging but doable)")
    print("  - SNR < 0 dB: Extreme noise (may miss speech)")
    print("  - Noise ONLY: Should ALWAYS reject (no false positives)")
    print()

    return accuracy


if __name__ == "__main__":
    accuracy = run_realistic_tests()
