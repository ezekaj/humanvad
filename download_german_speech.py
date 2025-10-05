"""
Download German speech samples from Mozilla Common Voice
For testing VAD with real German conversational speech
"""

import os
from datasets import load_dataset
import numpy as np
import soundfile as sf

def download_german_samples(num_samples=20, output_dir="test_audio"):
    """
    Download German speech samples from Common Voice

    Args:
        num_samples: Number of samples to download
        output_dir: Directory to save audio files
    """

    print("=" * 80)
    print("DOWNLOADING GERMAN SPEECH DATA FROM COMMON VOICE")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Common Voice German dataset (streaming mode)...")
    print("This will download a few samples without getting the full 1,040 hour dataset")
    print()

    # Load dataset in streaming mode (doesn't download everything)
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "de",
        split="test",
        streaming=True,
        trust_remote_code=True
    )

    print(f"Downloading {num_samples} samples...")
    print("-" * 80)

    samples_data = []

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        # Get audio data
        audio = sample['audio']
        audio_array = audio['array']
        sample_rate = audio['sampling_rate']

        # Get text
        text = sample['sentence']

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy import signal
            num_samples_new = int(len(audio_array) * 16000 / sample_rate)
            audio_array = signal.resample(audio_array, num_samples_new)
            sample_rate = 16000

        # Normalize to prevent clipping
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array)) * 0.9

        # Save audio file
        filename = f"german_sample_{i:03d}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio_array, sample_rate)

        # Store metadata
        samples_data.append({
            'filename': filename,
            'text': text,
            'duration': len(audio_array) / sample_rate,
            'sample_rate': sample_rate
        })

        print(f"  [{i+1:2d}/{num_samples}] {filename}: \"{text[:50]}{'...' if len(text) > 50 else ''}\" ({len(audio_array) / sample_rate:.2f}s)")

    print()
    print("-" * 80)
    print(f"Downloaded {len(samples_data)} samples to {output_dir}/")
    print()

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("Common Voice German Speech Samples\n")
        f.write("=" * 80 + "\n\n")
        for data in samples_data:
            f.write(f"File: {data['filename']}\n")
            f.write(f"Text: {data['text']}\n")
            f.write(f"Duration: {data['duration']:.2f}s\n")
            f.write(f"Sample Rate: {data['sample_rate']} Hz\n")
            f.write("-" * 80 + "\n")

    print(f"Metadata saved to {metadata_path}")
    print()

    # Print statistics
    total_duration = sum(d['duration'] for d in samples_data)
    avg_duration = total_duration / len(samples_data)

    print("STATISTICS:")
    print(f"  Total samples: {len(samples_data)}")
    print(f"  Total duration: {total_duration:.2f}s ({total_duration / 60:.2f} min)")
    print(f"  Average duration: {avg_duration:.2f}s")
    print(f"  Sample rate: 16000 Hz (mono)")
    print()
    print("=" * 80)

    return samples_data


if __name__ == "__main__":
    # Check if scipy is available (for resampling)
    try:
        import scipy
    except ImportError:
        print("Installing scipy for audio resampling...")
        import subprocess
        subprocess.check_call(["pip", "install", "scipy"])

    # Check if soundfile is available
    try:
        import soundfile
    except ImportError:
        print("Installing soundfile for audio I/O...")
        import subprocess
        subprocess.check_call(["pip", "install", "soundfile"])

    # Download samples
    samples = download_german_samples(num_samples=20, output_dir="test_audio")

    print()
    print("NEXT STEPS:")
    print("1. Check test_audio/ directory for downloaded WAV files")
    print("2. Update test harness to use real audio instead of random noise")
    print("3. Re-test baseline and features with real German speech")
    print()
