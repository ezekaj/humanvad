"""
Test VAD baseline and features with REAL German speech audio
Uses gTTS-generated samples instead of random noise
"""

import numpy as np
import soundfile as sf
import os
from excellence_vad_german import ExcellenceVADGerman


def load_audio_frames(filepath, frame_size=160, amplify=3.0):
    """
    Load audio file and yield frames

    Args:
        filepath: Path to WAV file
        frame_size: Samples per frame (160 = 10ms at 16kHz)
        amplify: Amplification factor (gTTS audio is often quiet)

    Yields:
        Audio frames as numpy arrays
    """
    audio, sr = sf.read(filepath)

    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}Hz")

    # Amplify audio (gTTS tends to be quiet for VAD detection)
    audio = audio * amplify

    # Clip to prevent distortion
    audio = np.clip(audio, -0.95, 0.95)

    # Pad to frame boundary
    remainder = len(audio) % frame_size
    if remainder != 0:
        audio = np.pad(audio, (0, frame_size - remainder))

    # Yield frames
    for i in range(0, len(audio), frame_size):
        yield audio[i:i + frame_size]


def test_baseline_with_real_audio():
    """
    Test baseline VAD with real German speech (gTTS-generated)
    """

    print("=" * 80)
    print(" BASELINE TEST WITH REAL GERMAN SPEECH")
    print("=" * 80)
    print()

    audio_dir = "test_audio"

    # Test scenarios (matching generated audio files)
    scenarios = [
        {
            'filename': 'complete_sentence_1.wav',
            'text': 'Das Hotel hat fünfzig Zimmer',
            'expected': 'interrupt',
            'description': 'Complete sentence (hotel rooms)'
        },
        {
            'filename': 'complete_sentence_2.wav',
            'text': 'Vielen Dank für Ihren Anruf',
            'expected': 'interrupt',
            'description': 'Complete sentence (thank you)'
        },
        {
            'filename': 'complete_sentence_3.wav',
            'text': 'Guten Tag, wie kann ich Ihnen helfen',
            'expected': 'interrupt',
            'description': 'Complete greeting'
        },
        {
            'filename': 'incomplete_hesitation.wav',
            'text': 'Ich möchte Ihnen sagen dass',
            'expected': 'wait',
            'description': 'Incomplete (ends with "dass")'
        },
        {
            'filename': 'incomplete_preposition.wav',
            'text': 'Ich gehe zur',
            'expected': 'wait',
            'description': 'Incomplete (ends with preposition)'
        },
        {
            'filename': 'complete_with_number.wav',
            'text': 'Der Preis beträgt zweihundert Euro',
            'expected': 'interrupt',
            'description': 'Complete with number'
        },
        {
            'filename': 'complete_confirmation.wav',
            'text': 'Ja, das ist korrekt',
            'expected': 'interrupt',
            'description': 'Complete confirmation'
        },
        {
            'filename': 'incomplete_conjunction.wav',
            'text': 'Das Zimmer ist verfügbar und',
            'expected': 'wait',
            'description': 'Incomplete (ends with "und")'
        },
        {
            'filename': 'complete_question.wav',
            'text': 'Haben Sie noch weitere Fragen',
            'expected': 'interrupt',
            'description': 'Complete question'
        },
        {
            'filename': 'complete_polite.wav',
            'text': 'Sehr gerne, ich helfe Ihnen',
            'expected': 'interrupt',
            'description': 'Complete polite response'
        },
    ]

    # Initialize VAD
    vad = ExcellenceVADGerman(use_adaptive_threshold=False)

    correct = 0
    results = []

    print("Testing with REAL German speech audio (gTTS)...")
    print("NOTE: First checking if audio is detected as speech by VAD...")
    print("-" * 80)

    for scenario in scenarios:
        filepath = os.path.join(audio_dir, scenario['filename'])

        if not os.path.exists(filepath):
            print(f"  SKIP {scenario['filename']:35} (file not found)")
            continue

        # Process audio file frame by frame
        # In real scenario, user speaks OVER AI speech (overlap)
        # We'll feed both user audio and AI audio simultaneously

        final_result = None
        user_frames = list(load_audio_frames(filepath))

        for i, user_frame in enumerate(user_frames):
            # Simulate AI also speaking (generate simple noise as placeholder)
            ai_frame = np.random.randn(160) * 0.1  # Low noise level

            # Process frame
            result = vad.process_frame(
                user_frame=user_frame,
                ai_frame=ai_frame,
                ai_text=scenario['text']
            )

            # Update final result
            final_result = result

        # Check if correct
        action = final_result['action']
        is_correct = action == scenario['expected']
        correct += is_correct

        status = "OK  " if is_correct else "FAIL"
        prob = final_result['turn_end_prob']

        print(f"  {status} {scenario['description']:40} prob={prob:.3f} -> {action:20}")

        results.append({
            'scenario': scenario,
            'result': final_result,
            'correct': is_correct
        })

    print()
    print("-" * 80)

    accuracy = (correct / len(scenarios)) * 100
    print(f"Baseline Accuracy: {accuracy:.0f}% ({correct}/{len(scenarios)})")
    print()

    # Show detailed results
    print("DETAILED RESULTS:")
    print("-" * 80)
    for r in results:
        scenario = r['scenario']
        result = r['result']
        correct_mark = "OK" if r['correct'] else "FAIL"

        print(f"{correct_mark} {scenario['description']}")
        print(f"   Text: \"{scenario['text']}\"")
        print(f"   Expected: {scenario['expected']}")
        print(f"   Got: {result['action']}")
        print(f"   Turn-end prob: {result['turn_end_prob']:.3f}")
        print(f"   User speaking: {result['user_speaking']}")
        print(f"   AI speaking: {result['ai_speaking']}")
        print()

    print("=" * 80)

    return accuracy, results


def test_feature7_with_real_audio():
    """
    Test Feature #7 (Adaptive Thresholds) with real German speech
    """

    print("=" * 80)
    print(" FEATURE #7 TEST WITH REAL GERMAN SPEECH")
    print("=" * 80)
    print()

    audio_dir = "test_audio"

    scenarios = [
        {
            'filename': 'complete_sentence_1.wav',
            'text': 'Das Hotel hat fünfzig Zimmer',
            'expected': 'interrupt',
            'description': 'Complete sentence (hotel rooms)'
        },
        {
            'filename': 'complete_sentence_2.wav',
            'text': 'Vielen Dank für Ihren Anruf',
            'expected': 'interrupt',
            'description': 'Complete sentence (thank you)'
        },
        {
            'filename': 'complete_sentence_3.wav',
            'text': 'Guten Tag, wie kann ich Ihnen helfen',
            'expected': 'interrupt',
            'description': 'Complete greeting'
        },
        {
            'filename': 'incomplete_hesitation.wav',
            'text': 'Ich möchte Ihnen sagen dass',
            'expected': 'wait',
            'description': 'Incomplete (ends with "dass")'
        },
        {
            'filename': 'incomplete_preposition.wav',
            'text': 'Ich gehe zur',
            'expected': 'wait',
            'description': 'Incomplete (ends with preposition)'
        },
        {
            'filename': 'complete_with_number.wav',
            'text': 'Der Preis beträgt zweihundert Euro',
            'expected': 'interrupt',
            'description': 'Complete with number'
        },
        {
            'filename': 'complete_confirmation.wav',
            'text': 'Ja, das ist korrekt',
            'expected': 'interrupt',
            'description': 'Complete confirmation'
        },
        {
            'filename': 'incomplete_conjunction.wav',
            'text': 'Das Zimmer ist verfügbar und',
            'expected': 'wait',
            'description': 'Incomplete (ends with "und")'
        },
        {
            'filename': 'complete_question.wav',
            'text': 'Haben Sie noch weitere Fragen',
            'expected': 'interrupt',
            'description': 'Complete question'
        },
        {
            'filename': 'complete_polite.wav',
            'text': 'Sehr gerne, ich helfe Ihnen',
            'expected': 'interrupt',
            'description': 'Complete polite response'
        },
    ]

    # Initialize VAD with Feature #7
    vad = ExcellenceVADGerman(use_adaptive_threshold=True)

    correct = 0
    results = []

    print("Testing Feature #7 with REAL German speech audio (gTTS)...")
    print("-" * 80)

    for scenario in scenarios:
        filepath = os.path.join(audio_dir, scenario['filename'])

        if not os.path.exists(filepath):
            print(f"  SKIP {scenario['filename']:35} (file not found)")
            continue

        final_result = None
        user_frames = list(load_audio_frames(filepath))

        for i, user_frame in enumerate(user_frames):
            ai_frame = np.random.randn(160) * 0.1

            result = vad.process_frame(
                user_frame=user_frame,
                ai_frame=ai_frame,
                ai_text=scenario['text']
            )

            final_result = result

        action = final_result['action']
        is_correct = action == scenario['expected']
        correct += is_correct

        status = "OK  " if is_correct else "FAIL"
        prob = final_result['turn_end_prob']
        thresh = final_result.get('adaptive_threshold', 0.60)

        print(f"  {status} {scenario['description']:40} prob={prob:.3f} thresh={thresh:.3f} -> {action:20}")

        results.append({
            'scenario': scenario,
            'result': final_result,
            'correct': is_correct
        })

    print()
    print("-" * 80)

    accuracy = (correct / len(scenarios)) * 100
    print(f"Feature #7 Accuracy: {accuracy:.0f}% ({correct}/{len(scenarios)})")
    print()
    print("=" * 80)

    return accuracy, results


if __name__ == "__main__":
    print()
    print("TESTING WITH REAL GERMAN SPEECH (gTTS-generated)")
    print()

    # Test baseline
    baseline_acc, baseline_results = test_baseline_with_real_audio()

    print()

    # Test Feature #7
    feature7_acc, feature7_results = test_feature7_with_real_audio()

    print()
    print("=" * 80)
    print(" COMPARISON")
    print("=" * 80)
    print(f"Baseline:   {baseline_acc:.0f}%")
    print(f"Feature #7: {feature7_acc:.0f}%")
    print(f"Improvement: {feature7_acc - baseline_acc:+.0f}%")
    print()

    if feature7_acc > baseline_acc:
        print("[SUCCESS] Feature #7 improves accuracy with real audio")
    elif feature7_acc == baseline_acc:
        print("[NO CHANGE] Feature #7 does not improve accuracy")
    else:
        print("[REGRESSION] Feature #7 decreases accuracy")

    print("=" * 80)
    print()
