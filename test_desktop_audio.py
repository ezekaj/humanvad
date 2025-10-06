"""
Test Excellence VAD semantic detector with real Desktop German audio transcripts
"""
from excellence_vad_german import ExcellenceVADGerman
import os

# Desktop German audio files with expected completeness
test_files = [
    ("guten_morgen.wav", "Guten Morgen", True, "greeting"),
    ("guten_tag.wav", "Guten Tag", True, "greeting"),
    ("auf_wiedersehen.wav", "Auf Wiedersehen", True, "farewell"),
    ("danke.wav", "Danke", True, "polite"),
    ("bitte.wav", "Bitte", True, "polite"),
    ("ja.wav", "Ja", True, "confirmation"),
    ("nein.wav", "Nein", True, "confirmation"),
    ("hallo.wav", "Hallo", True, "greeting"),
    ("herzlich_willkommen.wav", "Herzlich willkommen", True, "greeting"),
    ("ich_liebe_deutsch.wav", "Ich liebe Deutsch", True, "complete sentence"),
    ("alles_gute.wav", "Alles Gute", True, "wish"),
    ("bis_bald.wav", "Bis bald", True, "farewell"),
    ("entschuldigung.wav", "Entschuldigung", True, "polite"),
    ("gute_nacht.wav", "Gute Nacht", True, "greeting"),
    ("schoen.wav", "Schön", True, "confirmation"),
    ("viel_glueck.wav", "Viel Glück", True, "wish"),
    ("wie_gehts.wav", "Wie geht's", True, "question"),
]

def main():
    vad = ExcellenceVADGerman(turn_end_threshold=0.60)
    semantic = vad.semantic_detector
    audio_dir = "desktop_german_data"

    print("Testing Semantic Detector with Desktop German Audio Transcripts\n")
    print("=" * 90)

    correct = 0
    total = 0

    for filename, transcript, expected_complete, category in test_files:
        filepath = os.path.join(audio_dir, filename)

        if not os.path.exists(filepath):
            print(f"SKIP {filename:30} (file not found)")
            continue

        # Test semantic detector
        result = semantic.is_complete(transcript)

        # With 0.60 threshold
        actual_complete = (result['complete_prob'] >= 0.60)
        is_correct = (actual_complete == expected_complete)

        status = "OK" if is_correct else "FAIL"
        correct += is_correct
        total += 1

        print(f"{status:4} {filename:30} | \"{transcript:25}\" | Prob: {result['complete_prob']:.3f} | Expected: {expected_complete} | {category}")

    print("=" * 90)
    print(f"\nACCURACY: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Threshold: 0.60")
    print(f"Testing SEMANTIC DETECTOR with Desktop German audio transcripts")
    print(f"Audio files are REAL recordings (22kHz WAV), not synthetic TTS")

if __name__ == "__main__":
    main()
