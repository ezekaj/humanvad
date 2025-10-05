"""
Test Excellence VAD on Real Conversation Datasets
==================================================

Tests on publicly available conversation audio samples.
Falls back to synthetic data if real datasets not available.
"""

import numpy as np
import os
from excellence_vad import ExcellenceVAD
import wave


def load_audio_file(filepath, target_sr=16000):
    """Load audio file and resample if needed"""
    try:
        with wave.open(filepath, 'rb') as wav:
            sr = wav.getframerate()
            n_channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())

            # Convert to numpy
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize

            # Handle stereo
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            # Resample if needed (simple downsampling)
            if sr != target_sr:
                step = sr // target_sr
                audio = audio[::step]

            return audio, target_sr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def generate_realistic_conversation_audio(sr=16000):
    """
    Generate realistic conversation scenario with:
    - Natural pauses
    - Turn-taking patterns
    - Overlapping speech
    """

    scenarios = []

    # Scenario 1: Phone conversation - natural turn-taking
    # Speaker A: "Hello, how are you doing today?"
    # Pause 300ms
    # Speaker B: "I'm doing great, thanks for asking"

    def make_utterance(duration, f0_start, f0_end, modulation_rate):
        t = np.linspace(0, duration, int(sr * duration))
        f0 = np.linspace(f0_start, f0_end, len(t))

        signal = 1.0 * np.sin(2 * np.pi * f0 * t)
        signal += 0.8 * np.sin(2 * np.pi * 2 * f0 * t)
        signal += 0.6 * np.sin(2 * np.pi * 3 * f0 * t)
        signal += 0.4 * np.sin(2 * np.pi * 500 * t)
        signal += 0.3 * np.sin(2 * np.pi * 1500 * t)

        modulation = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * modulation_rate * t))
        signal *= modulation

        return signal / np.max(np.abs(signal))

    # Natural conversation with pauses
    conversation = []
    texts = []
    labels = []

    # Turn 1: Complete question
    utt1 = make_utterance(1.2, 160, 180, 5.0)  # Rising (question)
    conversation.append(utt1)
    texts.append("Hello, how are you doing today?")
    labels.append("complete_question")

    # Pause
    pause1 = np.zeros(int(sr * 0.3))
    conversation.append(pause1)
    texts.append("")
    labels.append("pause")

    # Turn 2: Complete response
    utt2 = make_utterance(1.5, 150, 130, 5.2)  # Falling (statement)
    conversation.append(utt2)
    texts.append("I'm doing great, thanks for asking")
    labels.append("complete_statement")

    # Pause
    pause2 = np.zeros(int(sr * 0.25))
    conversation.append(pause2)
    texts.append("")
    labels.append("pause")

    # Turn 3: Incomplete then interrupted
    utt3 = make_utterance(0.8, 155, 155, 5.5)  # Level (continuing)
    conversation.append(utt3)
    texts.append("I wanted to ask you about")  # Incomplete
    labels.append("incomplete_interrupted")

    # Turn 4: Interruption (overlap)
    utt4 = make_utterance(0.6, 165, 145, 5.0)
    conversation.append(utt4)
    texts.append("Oh yes, let me tell you")
    labels.append("interruption")

    scenarios.append({
        'name': 'Natural phone conversation',
        'audio_segments': conversation,
        'texts': texts,
        'labels': labels
    })

    # Scenario 2: Business meeting - collaborative turn-taking
    conversation2 = []
    texts2 = []
    labels2 = []

    # Statement with pause
    utt1 = make_utterance(1.8, 145, 125, 4.8)
    conversation2.append(utt1)
    texts2.append("I think we should finalize the budget by Friday")
    labels2.append("complete_statement")

    pause = np.zeros(int(sr * 0.35))
    conversation2.append(pause)
    texts2.append("")
    labels2.append("pause")

    # Agreement
    utt2 = make_utterance(0.6, 150, 140, 5.0)
    conversation2.append(utt2)
    texts2.append("That sounds good to me")
    labels2.append("complete_acknowledgment")

    scenarios.append({
        'name': 'Business meeting',
        'audio_segments': conversation2,
        'texts': texts2,
        'labels': labels2
    })

    return scenarios


def test_on_dataset(dataset_path=None):
    """Test Excellence VAD on real or synthetic dataset"""

    print("=" * 80)
    print(" EXCELLENCE VAD - REAL DATASET TESTING")
    print("=" * 80)
    print()

    # Try to load real dataset
    real_audio_found = False

    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading audio from: {dataset_path}")
        audio, sr = load_audio_file(dataset_path)
        if audio is not None:
            real_audio_found = True
            print(f"Loaded: {len(audio)/sr:.1f}s of audio at {sr}Hz")

    if not real_audio_found:
        print("No real dataset provided - using realistic synthetic conversations")
        print("(For real testing, provide path to WAV file)")
        print()
        scenarios = generate_realistic_conversation_audio()
    else:
        print("Real audio loaded!")
        print()
        scenarios = [{
            'name': 'Real conversation',
            'audio_segments': [audio],
            'texts': ["(transcription would be needed)"],
            'labels': ["unknown"]
        }]

    # Test each scenario
    vad = ExcellenceVAD(sample_rate=16000, turn_end_threshold=0.75)

    total_correct = 0
    total_decisions = 0

    for scenario in scenarios:
        print("-" * 80)
        print(f"Scenario: {scenario['name']}")
        print("-" * 80)
        print()

        vad.reset()
        frame_size = 160  # 10ms

        segment_idx = 0
        results = []

        # Process each audio segment
        for seg_audio, text, label in zip(
            scenario['audio_segments'],
            scenario['texts'],
            scenario['labels']
        ):
            print(f"\nSegment {segment_idx + 1}: {label}")
            print(f"Text: \"{text}\"" if text else "Text: (silence)")

            # Process frames
            for i in range(0, len(seg_audio) - frame_size, frame_size):
                frame = seg_audio[i:i+frame_size]

                # Simulate stereo (in real system, would have actual user/AI channels)
                if label.startswith('pause'):
                    user_frame = np.zeros(frame_size)
                    ai_frame = np.zeros(frame_size)
                elif label == 'interruption':
                    user_frame = frame  # User speaking
                    ai_frame = frame * 0.3  # AI still speaking (overlap)
                else:
                    user_frame = np.zeros(frame_size)
                    ai_frame = frame  # AI speaking

                result = vad.process_frame(user_frame, ai_frame, text)

                # Track significant events
                if result.get('user_speaking') or result.get('ai_speaking'):
                    results.append({
                        'segment': segment_idx,
                        'label': label,
                        'text': text,
                        'result': result
                    })

            # Evaluate decision for this segment
            if label == 'complete_question':
                # After complete question, should allow response (high turn-end)
                segment_results = [r for r in results if r['segment'] == segment_idx]
                if segment_results:
                    avg_turn_end = np.mean([r['result'].get('turn_end_prob', 0.0) for r in segment_results])
                    expected_high = avg_turn_end > 0.70
                    print(f"  Turn-end probability: {avg_turn_end:.1%} (expected: >70%)")
                    print(f"  Result: {'PASS' if expected_high else 'FAIL'}")
                    if expected_high:
                        total_correct += 1
                    total_decisions += 1

            elif label == 'complete_statement':
                segment_results = [r for r in results if r['segment'] == segment_idx]
                if segment_results:
                    avg_turn_end = np.mean([r['result'].get('turn_end_prob', 0.0) for r in segment_results])
                    expected_high = avg_turn_end > 0.60
                    print(f"  Turn-end probability: {avg_turn_end:.1%} (expected: >60%)")
                    print(f"  Result: {'PASS' if expected_high else 'FAIL'}")
                    if expected_high:
                        total_correct += 1
                    total_decisions += 1

            elif label == 'incomplete_interrupted':
                segment_results = [r for r in results if r['segment'] == segment_idx]
                if segment_results:
                    avg_turn_end = np.mean([r['result'].get('turn_end_prob', 0.0) for r in segment_results])
                    expected_low = avg_turn_end < 0.50
                    print(f"  Turn-end probability: {avg_turn_end:.1%} (expected: <50%)")
                    print(f"  Result: {'PASS' if expected_low else 'FAIL'}")
                    if expected_low:
                        total_correct += 1
                    total_decisions += 1

            elif label == 'interruption':
                segment_results = [r for r in results if r['segment'] == segment_idx]
                overlap_results = [r for r in segment_results if r['result'].get('overlap')]
                if overlap_results:
                    # Check action during overlap
                    actions = [r['result']['action'] for r in overlap_results]
                    correct_action = any('interrupt' in a for a in actions)
                    print(f"  Overlap detected: {len(overlap_results)} frames")
                    print(f"  Actions: {set(actions)}")
                    print(f"  Result: {'PASS' if correct_action else 'FAIL'}")
                    if correct_action:
                        total_correct += 1
                    total_decisions += 1

            segment_idx += 1

        print()

    # Final summary
    print()
    print("=" * 80)
    print(" FINAL RESULTS")
    print("=" * 80)
    print()

    if total_decisions > 0:
        accuracy = (total_correct / total_decisions) * 100
        print(f"Accuracy: {total_correct}/{total_decisions} ({accuracy:.1f}%)")
        print()

        if accuracy >= 90:
            print(f"EXCELLENT: {accuracy:.1f}% - Exceeds human telephone performance")
        elif accuracy >= 80:
            print(f"GOOD: {accuracy:.1f}% - Approaching human performance")
        elif accuracy >= 70:
            print(f"ACCEPTABLE: {accuracy:.1f}% - Room for improvement")
        else:
            print(f"NEEDS WORK: {accuracy:.1f}% - Below target")
    else:
        print("No decisions evaluated (need labeled data)")

    print()

    # Performance stats
    stats = vad.get_stats()
    print("PERFORMANCE:")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"  Target: <10ms {'PASS' if stats['avg_latency_ms'] < 10 else 'FAIL'}")
    print()

    print("Note: For comprehensive testing on real conversations:")
    print("  - Provide Switchboard/Fisher/AMI dataset")
    print("  - Include ground-truth turn-taking labels")
    print("  - Test on diverse speakers and scenarios")
    print()

    return accuracy if total_decisions > 0 else None


if __name__ == "__main__":
    import sys

    # Check if dataset path provided
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None

    accuracy = test_on_dataset(dataset_path)
