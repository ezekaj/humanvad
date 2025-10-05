"""
Generate German speech samples using Google Text-to-Speech (gTTS)
For testing VAD with real German speech instead of random noise
"""

import os
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
import tempfile

def generate_german_speech_samples(output_dir="test_audio"):
    """
    Generate German speech samples using gTTS

    Args:
        output_dir: Directory to save audio files
    """

    print("=" * 80)
    print("GENERATING GERMAN SPEECH SAMPLES WITH gTTS")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Test scenarios matching our VAD test cases
    scenarios = [
        {
            'name': 'complete_sentence_1',
            'text': 'Das Hotel hat fünfzig Zimmer',
            'description': 'Complete sentence (hotel rooms)',
            'expected': 'interrupt'
        },
        {
            'name': 'complete_sentence_2',
            'text': 'Vielen Dank für Ihren Anruf',
            'description': 'Complete sentence (thank you)',
            'expected': 'interrupt'
        },
        {
            'name': 'complete_sentence_3',
            'text': 'Guten Tag, wie kann ich Ihnen helfen',
            'description': 'Complete greeting',
            'expected': 'interrupt'
        },
        {
            'name': 'incomplete_hesitation',
            'text': 'Ich möchte Ihnen sagen dass',
            'description': 'Incomplete (ends with "dass")',
            'expected': 'wait'
        },
        {
            'name': 'incomplete_preposition',
            'text': 'Ich gehe zur',
            'description': 'Incomplete (ends with preposition)',
            'expected': 'wait'
        },
        {
            'name': 'complete_with_number',
            'text': 'Der Preis beträgt zweihundert Euro',
            'description': 'Complete with number',
            'expected': 'interrupt'
        },
        {
            'name': 'complete_confirmation',
            'text': 'Ja, das ist korrekt',
            'description': 'Complete confirmation',
            'expected': 'interrupt'
        },
        {
            'name': 'incomplete_conjunction',
            'text': 'Das Zimmer ist verfügbar und',
            'description': 'Incomplete (ends with "und")',
            'expected': 'wait'
        },
        {
            'name': 'complete_question',
            'text': 'Haben Sie noch weitere Fragen',
            'description': 'Complete question',
            'expected': 'interrupt'
        },
        {
            'name': 'complete_polite',
            'text': 'Sehr gerne, ich helfe Ihnen',
            'description': 'Complete polite response',
            'expected': 'interrupt'
        },
    ]

    print("Generating speech samples...")
    print("-" * 80)

    samples_data = []

    for i, scenario in enumerate(scenarios):
        text = scenario['text']
        filename = f"{scenario['name']}.wav"
        filepath = os.path.join(output_dir, filename)

        print(f"  [{i+1:2d}/{len(scenarios)}] {scenario['name']:30} \"{text}\"")

        # Generate speech with gTTS
        tts = gTTS(text=text, lang='de', slow=False)

        # Save to temporary MP3
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tts.save(tmp_path)

        # Convert MP3 to 16kHz mono WAV using pydub
        audio = AudioSegment.from_mp3(tmp_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        audio.export(filepath, format='wav')

        # Clean up temp file
        os.remove(tmp_path)

        # Get duration
        duration = len(audio) / 1000.0  # pydub uses milliseconds

        samples_data.append({
            'filename': filename,
            'text': text,
            'description': scenario['description'],
            'expected_action': scenario['expected'],
            'duration': duration
        })

    print()
    print("-" * 80)
    print(f"Generated {len(samples_data)} samples in {output_dir}/")
    print()

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("German Speech Test Samples (gTTS)\n")
        f.write("=" * 80 + "\n\n")
        for data in samples_data:
            f.write(f"File: {data['filename']}\n")
            f.write(f"Text: {data['text']}\n")
            f.write(f"Description: {data['description']}\n")
            f.write(f"Expected Action: {data['expected_action']}\n")
            f.write(f"Duration: {data['duration']:.2f}s\n")
            f.write("-" * 80 + "\n")

    print(f"Metadata saved to {metadata_path}")
    print()

    # Print statistics
    total_duration = sum(d['duration'] for d in samples_data)
    avg_duration = total_duration / len(samples_data)

    complete_count = sum(1 for d in samples_data if d['expected_action'] == 'interrupt')
    incomplete_count = sum(1 for d in samples_data if d['expected_action'] == 'wait')

    print("STATISTICS:")
    print(f"  Total samples: {len(samples_data)}")
    print(f"  Complete sentences (interrupt): {complete_count}")
    print(f"  Incomplete sentences (wait): {incomplete_count}")
    print(f"  Total duration: {total_duration:.2f}s")
    print(f"  Average duration: {avg_duration:.2f}s")
    print(f"  Sample rate: 16000 Hz (mono)")
    print()
    print("=" * 80)

    return samples_data


if __name__ == "__main__":
    # Check dependencies
    try:
        from gtts import gTTS
    except ImportError:
        print("Installing gTTS...")
        import subprocess
        subprocess.check_call(["pip", "install", "gTTS"])

    try:
        from pydub import AudioSegment
    except ImportError:
        print("Installing pydub...")
        import subprocess
        subprocess.check_call(["pip", "install", "pydub"])

    # Generate samples
    samples = generate_german_speech_samples(output_dir="test_audio")

    print()
    print("NEXT STEPS:")
    print("1. Check test_audio/ directory for generated WAV files")
    print("2. Update test harness to load real audio instead of random noise")
    print("3. Re-test baseline and features with real German speech")
    print()
