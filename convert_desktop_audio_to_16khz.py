"""
Convert Desktop German audio from 22kHz to 16kHz for VAD testing
"""
import os
import wave
import numpy as np
from scipy import signal

input_dir = "desktop_german_data"
output_dir = "desktop_german_data_16khz"

os.makedirs(output_dir, exist_ok=True)

files = [
    "guten_morgen.wav",
    "guten_tag.wav",
    "auf_wiedersehen.wav",
    "danke.wav",
    "bitte.wav",
    "ja.wav",
    "nein.wav",
    "hallo.wav",
    "herzlich_willkommen.wav",
    "ich_liebe_deutsch.wav",
    "alles_gute.wav",
    "bis_bald.wav",
    "entschuldigung.wav",
    "gute_nacht.wav",
    "schoen.wav",
    "ueber.wav",
    "viel_glueck.wav",
    "wie_gehts.wav",
]

print("Converting Desktop German audio from 22kHz to 16kHz...\n")

for filename in files:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(input_path):
        print(f"SKIP {filename:30} (not found)")
        continue

    # Load 22kHz audio
    with wave.open(input_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)

    # Resample 22050 -> 16000
    num_samples = int(len(audio) * 16000 / sample_rate)
    resampled = signal.resample(audio, num_samples)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

    # Save as 16kHz mono WAV
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        wf.writeframes(resampled.tobytes())

    print(f"OK {filename:30} ({sample_rate}Hz -> 16000Hz)")

print(f"\nConverted {len(files)} files to {output_dir}/")
print("Format: 16kHz, mono, WAV (for Excellence VAD)")
