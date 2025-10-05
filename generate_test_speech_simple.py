"""
Generate German speech samples using Google Text-to-Speech (gTTS)
Simplified version - saves MP3 and provides instructions for conversion
"""

import os
from gtts import gTTS

def generate_german_speech_samples(output_dir="test_audio_mp3"):
    """
    Generate German speech samples using gTTS (MP3 format)
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
        filename = f"{scenario['name']}.mp3"
        filepath = os.path.join(output_dir, filename)

        print(f"  [{i+1:2d}/{len(scenarios)}] {scenario['name']:30} \"{text}\"")

        # Generate speech with gTTS
        tts = gTTS(text=text, lang='de', slow=False)
        tts.save(filepath)

        samples_data.append({
            'filename': filename,
            'text': text,
            'description': scenario['description'],
            'expected_action': scenario['expected']
        })

    print()
    print("-" * 80)
    print(f"Generated {len(samples_data)} MP3 samples in {output_dir}/")
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
            f.write("-" * 80 + "\n")

    print(f"Metadata saved to {metadata_path}")
    print()

    # Print statistics
    complete_count = sum(1 for d in samples_data if d['expected_action'] == 'interrupt')
    incomplete_count = sum(1 for d in samples_data if d['expected_action'] == 'wait')

    print("STATISTICS:")
    print(f"  Total samples: {len(samples_data)}")
    print(f"  Complete sentences (interrupt): {complete_count}")
    print(f"  Incomplete sentences (wait): {incomplete_count}")
    print()
    print("=" * 80)
    print()
    print("CONVERSION TO WAV (16kHz):")
    print()
    print("Option 1 - Using ffmpeg (if installed):")
    print(f"  cd {output_dir}")
    print("  for file in *.mp3; do")
    print('    ffmpeg -i "$file" -ar 16000 -ac 1 "${file%.mp3}.wav"')
    print("  done")
    print()
    print("Option 2 - Using online converter:")
    print("  Upload MP3 files to https://online-audio-converter.com/")
    print("  Select: WAV, 16000 Hz, Mono")
    print()
    print("Option 3 - Install ffmpeg:")
    print("  Windows: choco install ffmpeg")
    print("  Or download from: https://ffmpeg.org/download.html")
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

    # Generate samples
    samples = generate_german_speech_samples(output_dir="test_audio_mp3")

    print()
    print("NEXT STEPS:")
    print("1. Convert MP3 files to 16kHz WAV format (see instructions above)")
    print("2. Move WAV files to test_audio/ directory")
    print("3. Update test harness to load real audio instead of random noise")
    print("4. Re-test baseline and features with real German speech")
    print()
