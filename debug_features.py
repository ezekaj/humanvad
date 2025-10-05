"""
Debug: Check what features our synthetic speech actually has
"""

import numpy as np
from production_vad import ProductionVAD

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

# Generate clean speech
sr = 16000
speech = generate_speech_like(duration=0.5, sr=sr)

# Test features
vad = ProductionVAD(sample_rate=sr)
frame_size = 480  # 30ms
frame = speech[:frame_size]

features = vad._compute_features(frame)

print("=" * 80)
print(" SYNTHETIC SPEECH FEATURE ANALYSIS")
print("=" * 80)
print()
print(f"RMS Energy:         {features['rms']:.4f}")
print(f"Spectral Centroid:  {features['centroid']:.1f} Hz  (target: 400-3000 Hz)")
print(f"Spectral Rolloff:   {features['rolloff']:.1f} Hz  (target: 1500-6000 Hz)")
print(f"Zero-crossing Rate: {features['zcr']:.1f} Hz  (target: 40-250 Hz)")
print(f"Low-freq Ratio:     {features['lf_ratio']:.2%}  (target: 50-90%)")
print(f"Spectral Flux:      {features['flux']:.2f}")
print()

# Classify
confidence = vad._classify_speech(features)
print(f"Classification Confidence: {confidence:.1%}")
print(f"Detected as speech: {confidence > 0.45}")
print()

# Check gating conditions
snr = features['rms'] / (vad.noise_energy + 1e-6)
energy_score = min(snr / 2.5, 1.0)

centroid = features['centroid']
if 400 < centroid < 3000:
    centroid_score = 1.0 - abs(centroid - 1200) / 1200
else:
    centroid_score = 0.0
centroid_score = max(0, min(centroid_score, 1.0))

rolloff = features['rolloff']
if 1500 < rolloff < 6000:
    rolloff_score = 1.0 - abs(rolloff - 3500) / 3500
else:
    rolloff_score = 0.0
rolloff_score = max(0, min(rolloff_score, 1.0))

print("GATING SCORES:")
print(f"  Energy score:   {energy_score:.2f} (need > 0.3)")
print(f"  Centroid score: {centroid_score:.2f} (need > 0.2)")
print(f"  Rolloff score:  {rolloff_score:.2f} (need > 0.2)")
print()

gates_pass = (centroid_score > 0.2 and rolloff_score > 0.2 and energy_score > 0.3)
print(f"GATES PASS: {gates_pass}")
print()
