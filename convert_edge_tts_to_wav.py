"""
Convert Edge TTS MP3 files (misnamed as .wav) to actual WAV format
"""
import os
from pydub import AudioSegment

input_dir = "real_audio"
output_dir = "real_audio_converted"

os.makedirs(output_dir, exist_ok=True)

files = [
    "complete_sentence_1.wav",
    "complete_sentence_2.wav",
    "complete_sentence_3.wav",
    "incomplete_hesitation.wav",
    "incomplete_preposition.wav",
    "complete_with_number.wav",
    "complete_confirmation.wav",
    "incomplete_conjunction.wav",
    "complete_question.wav",
    "complete_polite.wav",
]

print("Converting Edge TTS files from MP3 to WAV...\n")

for filename in files:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Load MP3 (even though it's named .wav)
    audio = AudioSegment.from_file(input_path, format="mp3")

    # Convert to 16kHz mono WAV
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Export as WAV
    audio.export(output_path, format="wav")

    print(f"OK {filename}")

print(f"\nConverted {len(files)} files to {output_dir}/")
print("Format: 16kHz, mono, WAV")
