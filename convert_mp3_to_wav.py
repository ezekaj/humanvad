"""
Convert MP3 files to 16kHz WAV format using Python libraries
No ffmpeg required - uses pydub with built-in decoder
"""

import os
import sys

def convert_mp3_to_wav(input_dir="test_audio_mp3", output_dir="test_audio"):
    """
    Convert all MP3 files in input_dir to 16kHz WAV files in output_dir
    """

    print("=" * 80)
    print("CONVERTING MP3 TO WAV (16kHz MONO)")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all MP3 files
    mp3_files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]

    if not mp3_files:
        print(f"No MP3 files found in {input_dir}/")
        return []

    print(f"Found {len(mp3_files)} MP3 files")
    print("-" * 80)

    converted = []

    # Try different methods
    conversion_method = None

    # Method 1: Try moviepy (has built-in MP3 decoder)
    try:
        from moviepy.editor import AudioFileClip
        import numpy as np
        import wave

        conversion_method = "moviepy"
        print(f"Using moviepy for conversion...")
        print()

        for i, mp3_file in enumerate(mp3_files):
            mp3_path = os.path.join(input_dir, mp3_file)
            wav_file = mp3_file.replace('.mp3', '.wav')
            wav_path = os.path.join(output_dir, wav_file)

            print(f"  [{i+1:2d}/{len(mp3_files)}] {mp3_file} -> {wav_file}")

            # Load MP3
            audio = AudioFileClip(mp3_path)

            # Get audio data
            audio_array = audio.to_soundarray(fps=16000)  # 16kHz

            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            # Normalize to int16
            audio_int16 = (audio_array * 32767).astype(np.int16)

            # Save as WAV
            with wave.open(wav_path, 'w') as wav_file_obj:
                wav_file_obj.setnchannels(1)  # Mono
                wav_file_obj.setsampwidth(2)  # 16-bit
                wav_file_obj.setframerate(16000)  # 16kHz
                wav_file_obj.writeframes(audio_int16.tobytes())

            audio.close()
            converted.append(wav_file)

    except ImportError:
        print("moviepy not available, trying alternative method...")
        print()

        # Method 2: Try using just scipy and reading MP3 as raw audio
        try:
            import librosa
            import soundfile as sf

            conversion_method = "librosa"
            print(f"Using librosa for conversion...")
            print()

            for i, mp3_file in enumerate(mp3_files):
                mp3_path = os.path.join(input_dir, mp3_file)
                wav_file = mp3_file.replace('.mp3', '.wav')
                wav_path = os.path.join(output_dir, wav_file)

                print(f"  [{i+1:2d}/{len(mp3_files)}] {mp3_file} -> {wav_file}")

                # Load MP3 and resample to 16kHz
                audio, sr = librosa.load(mp3_path, sr=16000, mono=True)

                # Save as WAV
                sf.write(wav_path, audio, 16000)

                converted.append(wav_file)

        except ImportError:
            print("ERROR: No suitable audio library found!")
            print()
            print("Please install one of the following:")
            print("  pip install moviepy")
            print("  pip install librosa soundfile")
            print()
            print("Or install ffmpeg and use the shell command:")
            print("  choco install ffmpeg")
            return []

    if not conversion_method:
        return []

    print()
    print("-" * 80)
    print(f"Converted {len(converted)} files to {output_dir}/")
    print(f"Format: 16kHz, 16-bit, Mono WAV")
    print()
    print("=" * 80)

    return converted


if __name__ == "__main__":
    # Check and install dependencies
    try:
        import librosa
        import soundfile
        print("Using librosa + soundfile")
    except ImportError:
        try:
            import moviepy
            import numpy
            print("Using moviepy + numpy")
        except ImportError:
            print("Installing librosa and soundfile...")
            import subprocess
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa", "soundfile"])
                print("Installed successfully!")
                print()
            except Exception as e:
                print(f"Installation failed: {e}")
                print()
                print("Please manually install:")
                print("  pip install librosa soundfile")
                sys.exit(1)

    # Convert files
    converted = convert_mp3_to_wav()

    if converted:
        print()
        print("SUCCESS! Audio files ready for testing.")
        print()
        print("NEXT STEPS:")
        print("1. Check test_audio/ directory for WAV files")
        print("2. Run: python test_with_real_audio.py")
        print()
    else:
        print()
        print("Conversion failed. See error messages above.")
        print()
